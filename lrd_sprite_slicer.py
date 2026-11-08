import argparse
import json
import math
import os
import queue
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    scale: int = 4
    tolerance: float = 1.0
    spill: float = 1.0
    sharpen: float = 0.6
    denoise: float = 2.0
    feather: float = 0.5
    refine: int = 2
    margin: int = 8
    canvas: str = "tight"
    name: str = ""
    labels: tuple = ()
    ocr: str = "auto"
    bg: str = "auto"
    attach: float = 0.16
    workers: int = 0
    debug: bool = True
    review: bool = True
    max_side: int = 4096


def say(log, message):
    if log:
        log(message)
    else:
        print(message, flush=True)


def ellipse(radius):
    radius = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def slugify(text, fallback="item"):
    text = re.sub(r"\(.*", "", text or "")
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", text) if t]
    if len(tokens) > 1:
        tokens = [t for t in tokens if len(t) > 1] or tokens
    text = "_".join(tokens).lower()
    return text[:40] if text else fallback


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = None
    if img.shape[2] == 4:
        a = img[:, :, 3]
        if a.min() < 250 and float((a < 250).mean()) > 0.01:
            alpha = np.ascontiguousarray(a)
        img = img[:, :, :3]
    return np.ascontiguousarray(img), alpha


def write_png(path, bgra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError(f"Could not encode {path}")
    buf.tofile(path)


def estimate_background(bgr):
    h, w = bgr.shape[:2]
    t = max(2, min(h, w) // 50)
    border = np.concatenate([
        bgr[:t].reshape(-1, 3), bgr[-t:].reshape(-1, 3),
        bgr[:, :t].reshape(-1, 3), bgr[:, -t:].reshape(-1, 3),
    ])
    hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    greenish = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 120) & (hsv[:, 2] >= 100)
    pool = border[greenish] if greenish.sum() > 50 else border
    return np.median(pool, axis=0).astype(np.float32)


def parse_bg(text):
    if not text or text.lower() == "auto":
        return None
    parts = [float(v) for v in text.split(",")]
    return np.array(parts[::-1], np.float32)


def lab_distance(bgr, bg):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2LAB)
    ref = ref.astype(np.float32).reshape(3)
    dl = (lab[..., 0] - ref[0]) * (100.0 / 255.0)
    da = lab[..., 1] - ref[1]
    db = lab[..., 2] - ref[2]
    return np.sqrt(0.5 * dl * dl + da * da + db * db)


def key_info(bg):
    ch = int(np.argmax(bg))
    others = [i for i in range(3) if i != ch]
    strength = float(bg[ch] - max(bg[others[0]], bg[others[1]]))
    return ch, others[0], others[1], strength


def accepted_background(candidate, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros(candidate.shape, bool)
    border_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.zeros(n, bool)
    keep[border_labels] = True
    keep[stats[:, cv2.CC_STAT_AREA] >= min_area] = True
    keep[0] = False
    return keep[labels]


def propagate_colors(img, mask):
    h, w = img.shape[:2]
    m = mask.astype(np.float32)
    weighted = img * m[..., None]
    known = mask.copy()
    result = img.copy()
    k = 3
    limit = max(h, w) * 2
    while k <= limit and not known.all():
        num = cv2.boxFilter(weighted, -1, (k, k), borderType=cv2.BORDER_CONSTANT)
        den = cv2.boxFilter(m, -1, (k, k), borderType=cv2.BORDER_CONSTANT)
        fill = (~known) & (den > 1e-4)
        if fill.any():
            result[fill] = num[fill] / den[fill][:, None]
            known |= fill
        k = k * 2 + 1
    return result


def guided_filter(guide, src, radius, eps):
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    corr_ip = cv2.boxFilter(guide * src, -1, k)
    corr_ii = cv2.boxFilter(guide * guide, -1, k)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def chroma_alpha(bgr, bg, tolerance, scale):
    h, w = bgr.shape[:2]
    img = bgr.astype(np.float32)
    d = lab_distance(bgr, bg)
    tol = max(tolerance, 0.1)
    low, high = 12.0 * tol, 42.0 * tol
    candidate = d < low
    accepted = accepted_background(candidate, max(64, int(0.0008 * h * w)))
    dist_bg = cv2.distanceTransform((~accepted).astype(np.uint8), cv2.DIST_L2, 3)
    band = max(2.0, 2.5 * scale)
    reach = max(band + 1.0, 0.10 * min(h, w))
    core = (~accepted) & (d >= high) & (dist_bg > band)
    unknown = (~accepted) & (~core) & (dist_bg <= reach)
    alpha = np.ones((h, w), np.float32)
    alpha[accepted] = 0.0
    if not core.any():
        core = (~accepted) & (d >= high)
    if not core.any():
        return np.zeros((h, w), np.float32), img.copy()
    fg_est = propagate_colors(img, core)
    ch, o0, o1, strength = key_info(bg)
    if strength >= 60:
        e_img = img[..., ch] - np.maximum(img[..., o0], img[..., o1])
        e_fg = np.maximum(fg_est[..., ch] - np.maximum(fg_est[..., o0], fg_est[..., o1]), 0.0)
        denom = np.maximum(strength - e_fg, 20.0)
        a = np.clip((strength - e_img) / denom, 0.0, 1.0)
    else:
        a = np.clip((d - low) / (high - low), 0.0, 1.0)
    alpha[unknown] = a[unknown]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined = guided_filter(gray, alpha, max(2, int(round(1.5 * scale))), 1e-3)
    alpha = np.where(unknown, np.clip(refined, 0.0, 1.0), alpha)
    return alpha.astype(np.float32), fg_est


def estimate_foreground(img, alpha, bg, fg_est):
    a = alpha[..., None]
    unmix = np.clip((img - (1.0 - a) * bg) / np.maximum(a, 0.05), 0, 255)
    trust = np.clip((alpha - 0.15) / 0.35, 0.0, 1.0)[..., None]
    edge = trust * unmix + (1.0 - trust) * fg_est
    solid = np.clip((alpha - 0.9) / 0.08, 0.0, 1.0)[..., None]
    return solid * img + (1.0 - solid) * edge


def despill(color, alpha, bg, strength, scale):
    ch, o0, o1, key_strength = key_info(bg)
    if strength <= 0 or key_strength < 60:
        return color
    solid = (alpha > 0.6).astype(np.uint8)
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
    c0, c1 = color[..., o0], color[..., o1]
    excess_all = color[..., ch] - np.maximum(c0, c1)
    greenish = ((excess_all > 25) & (alpha > 0.6)).astype(np.uint8)
    thick = cv2.morphologyEx(greenish, cv2.MORPH_OPEN, ellipse(9.0 * scale))
    protect = cv2.dilate(thick, ellipse(2.0 * scale)).astype(np.float32)
    protect = np.clip(cv2.GaussianBlur(protect, (0, 0), 2.0 * scale), 0.0, 1.0)
    radius = 12.0 * scale
    weight = np.clip(1.0 - (dist - 1.0) / radius, 0.0, 1.0)
    weight = np.maximum(weight, np.clip((0.98 - alpha) * 4.0, 0.0, 1.0))
    weight = weight * (1.0 - protect)
    limit = 0.5 * (np.maximum(c0, c1) + 0.5 * (c0 + c1))
    excess = np.maximum(color[..., ch] - limit, 0.0)
    out = color.copy()
    out[..., ch] = color[..., ch] - excess * weight * min(strength, 1.5)
    return out


def sharpen_layer(color, amount, scale):
    if amount <= 0:
        return color
    sigma = 0.9 + 0.3 * scale
    blur = cv2.GaussianBlur(color, (0, 0), sigma)
    return np.clip(color + amount * (color - blur), 0, 255)


def load_superres(scale):
    if not hasattr(cv2, "dnn_superres"):
        return None
    for name in ("EDSR", "FSRCNN", "ESPCN", "LapSRN"):
        for folder in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "models"), os.getcwd()):
            path = os.path.join(folder, f"{name}_x{scale}.pb")
            if os.path.exists(path):
                try:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    sr.readModel(path)
                    sr.setModel(name.lower(), scale)
                    return sr
                except Exception:
                    continue
    return None


def upscale_hq(img, scale, iterations):
    if scale <= 1:
        return img.astype(np.float32)
    h, w = img.shape[:2]
    src = img.astype(np.float32)
    sr = load_superres(scale)
    if sr is not None:
        try:
            up = sr.upsample(np.clip(src, 0, 255).astype(np.uint8)).astype(np.float32)
            return up
        except Exception:
            pass
    size = (w * scale, h * scale)
    up = cv2.resize(src, size, interpolation=cv2.INTER_LANCZOS4)
    for _ in range(max(0, iterations)):
        down = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
        err = src - down
        up = up + cv2.resize(err, size, interpolation=cv2.INTER_CUBIC)
    return np.clip(up, 0, 255)


def fit_scale(h, w, scale, max_side):
    scale = max(1, int(scale))
    while scale > 1 and max(h, w) * scale > max_side:
        scale -= 1
    return scale


def foreground_mask(bgr, alpha, cfg):
    if alpha is not None:
        return alpha > 12, alpha >= 200, None, np.zeros(alpha.shape, np.float32)
    bg = parse_bg(cfg.bg)
    if bg is None:
        bg = estimate_background(bgr)
    d = lab_distance(bgr, bg)
    tol = max(cfg.tolerance, 0.1)
    return d >= 12.0 * tol, d >= 42.0 * tol, bg, d


def box_of(c):
    return (c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"])


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def box_overlap_ratio(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, smaller)


class DisjointSet:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_text(cands, h_ref):
    ds = DisjointSet(len(cands))
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            gap = max(0, max(a["x"], b["x"]) - min(a["x"] + a["w"], b["x"] + b["w"]))
            if gap <= 0.9 * h_ref and abs(a["cy"] - b["cy"]) <= 0.7 * h_ref:
                ds.union(i, j)
    groups = {}
    for i in range(len(cands)):
        groups.setdefault(ds.find(i), []).append(cands[i])
    return list(groups.values())


def symmetry(lab, c):
    x0, y0, x1, y1 = box_of(c)
    roi = lab[y0:y1, x0:x1] == c["id"]
    lr = roi & roi[:, ::-1]
    ud = roi & roi[::-1, :]
    return (lr.sum() / max(1, (roi | roi[:, ::-1]).sum()),
            ud.sum() / max(1, (roi | roi[::-1, :]).sum()))


def detect_text(bgr, mask, d, W, H):
    mn = bgr.min(axis=2).astype(np.int16)
    mx = bgr.max(axis=2).astype(np.int16)
    pure = ((mn >= 185) & ((mx - mn) <= 55)).astype(np.uint8)
    n, plab, st, ce = cv2.connectedComponentsWithStats(pure, connectivity=8)
    cands = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if 6 <= h <= 0.045 * H and w <= 0.05 * W and a >= 6:
            q = 4
            ya, yb, xa, xb = max(0, y - q), min(H, y + h + q), max(0, x - q), min(W, x + w + q)
            ring = np.ones((yb - ya, xb - xa), bool)
            ring[y - ya:y - ya + h, x - xa:x - xa + w] = False
            if float((~mask[ya:yb, xa:xb][ring]).mean()) < 0.05:
                continue
            cands.append({"id": i, "x": x, "y": y, "w": w, "h": h, "area": a, "cx": float(ce[i][0]),
                          "cy": y + h / 2.0})
    found = []
    if len(cands) < 2:
        return found, np.zeros((H, W), bool)
    ds = DisjointSet(len(cands))
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            hh = max(a["h"], b["h"])
            gap = max(0, max(a["x"], b["x"]) - min(a["x"] + a["w"], b["x"] + b["w"]))
            lo = min(a["h"], b["h"])
            if gap <= 1.7 * hh and abs(a["cy"] - b["cy"]) <= 0.45 * hh and (lo >= 0.6 * hh or lo <= 0.4 * hh):
                ds.union(i, j)
    groups = {}
    for i in range(len(cands)):
        groups.setdefault(ds.find(i), []).append(cands[i])
    text_px = np.zeros((H, W), bool)
    for group in groups.values():
        if len(group) < 2:
            continue
        box = union_box([box_of(c) for c in group])
        bw, bh = box[2] - box[0], box[3] - box[1]
        hs = np.array([c["h"] for c in group], np.float32)
        med = float(np.median(hs))
        steady = float((np.abs(hs - med) <= 0.3 * med).mean())
        min_ratio = 1.5 if len(group) >= 3 else 2.0
        if bw < min_ratio * bh or med < 7 or steady < 0.5:
            continue
        p = 4
        x0, y0, x1, y1 = max(0, box[0] - p), max(0, box[1] - p), min(W, box[2] + p), min(H, box[3] + p)
        ring = np.ones((y1 - y0, x1 - x0), bool)
        ring[2:-2, 2:-2] = False
        bgfrac = float((~mask[y0:y1, x0:x1][ring]).mean()) if ring.any() else 0.0
        if bgfrac < 0.55:
            continue
        sub = np.zeros((H, W), bool)
        for c in group:
            sub[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] |= plab[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] == c["id"]
        sub = cv2.dilate(sub.astype(np.uint8), ellipse(2)) > 0
        text_px |= sub
        found.append({"kind": "label", "box": (box[0], box[1], box[2], box[3]), "px": sub,
                      "cx": (box[0] + box[2]) / 2.0, "cy": (box[1] + box[3]) / 2.0, "text": "",
                      "area": int(sub.sum())})
    return found, text_px


def split_merged(lab, comps, core, mask, H, W, factor):
    core_u8 = (core & mask).astype(np.uint8)
    grown = cv2.dilate(core_u8, ellipse(2)) & mask.astype(np.uint8)
    kn, klab = cv2.connectedComponents(grown, connectivity=8)
    core_area = np.bincount(klab[core_u8 > 0], minlength=kn)
    best = []
    for c in comps.values():
        if c["h"] < 0.07 * H:
            continue
        x0, y0, x1, y1 = box_of(c)
        sub = (lab[y0:y1, x0:x1] == c["id"]) & (klab[y0:y1, x0:x1] > 0)
        ids = np.unique(klab[y0:y1, x0:x1][sub])
        best.append(max([core_area[k] for k in ids], default=0))
    if not best:
        return
    ref = float(np.median(best))
    widths = [c["w"] for c in comps.values() if c["h"] >= 0.07 * H]
    wide = factor * float(np.median(widths))
    next_id = int(lab.max()) + 1
    for cid in list(comps.keys()):
        c = comps[cid]
        if c["h"] < 0.07 * H or c["w"] < wide:
            continue
        x0, y0, x1, y1 = box_of(c)
        own = lab[y0:y1, x0:x1] == cid
        ids = [k for k in np.unique(klab[y0:y1, x0:x1][own & (klab[y0:y1, x0:x1] > 0)])
               if core_area[k] >= 0.45 * ref]
        if len(ids) < 2:
            continue
        dts = []
        for k in ids:
            marker = (klab[y0:y1, x0:x1] == k) & own
            dts.append(cv2.distanceTransform((~marker).astype(np.uint8), cv2.DIST_L2, 3))
        choice = np.argmin(np.stack(dts), axis=0)
        del comps[cid]
        for j in range(len(ids)):
            nid = cid if j == 0 else next_id
            if j:
                next_id += 1
            part = own & (choice == j)
            if not part.any():
                continue
            lab[y0:y1, x0:x1][own & (choice == j)] = nid
            ys, xs = np.nonzero(part)
            comps[nid] = {"id": nid, "x": int(x0 + xs.min()), "y": int(y0 + ys.min()),
                          "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1),
                          "area": int(part.sum()), "cx": float(x0 + xs.mean()), "cy": float(y0 + ys.mean()),
                          "kind": "pending"}


def valley_cut(own):
    total = int(own.sum())
    best = None
    for axis in (1, 0):
        prof = own.sum(axis=axis).astype(np.float64)
        n = len(prof)
        lo, hi = int(0.15 * n), int(0.85 * n)
        for i in range(lo, hi):
            left, right = prof[:i].max(), prof[i + 1:].max()
            ratio = prof[i] / max(1.0, min(left, right))
            if ratio <= 0.35 and (best is None or ratio < best[0]):
                part = own[:i].sum() if axis == 1 else own[:, :i].sum()
                if 0.12 * total <= part <= 0.88 * total:
                    best = (ratio, axis, i)
    if best is None:
        return None
    _, axis, i = best
    choice = np.zeros(own.shape, np.int32)
    if axis == 1:
        choice[i:, :] = 1
    else:
        choice[:, i:] = 1
    return choice


def split_necks(lab, comps, H):
    widths = [c["w"] for c in comps.values() if c["h"] >= 0.07 * H]
    if not widths:
        return
    wide = 2.2 * float(np.median(widths))
    next_id = int(lab.max()) + 1
    for cid in list(comps.keys()):
        c = comps[cid]
        if c["h"] < 0.07 * H or c["w"] < wide:
            continue
        x0, y0, x1, y1 = box_of(c)
        own = (lab[y0:y1, x0:x1] == cid).astype(np.uint8)
        total = int(own.sum())
        markers = None
        for r in range(2, 15):
            er = cv2.erode(own, ellipse(r))
            k, el, est, _ = cv2.connectedComponentsWithStats(er, connectivity=8)
            big = [i for i in range(1, k) if est[i, cv2.CC_STAT_AREA] >= 0.12 * total]
            if len(big) >= 2:
                markers = [el == i for i in big]
                break
            if k <= 1:
                break
        if markers is None:
            choice = valley_cut(own)
            if choice is None:
                continue
            markers = [None, None]
        else:
            dts = [cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for m in markers]
            choice = np.argmin(np.stack(dts), axis=0)
        del comps[cid]
        for j in range(len(markers)):
            nid = cid if j == 0 else next_id
            if j:
                next_id += 1
            part = (own > 0) & (choice == j)
            if not part.any():
                continue
            lab[y0:y1, x0:x1][part] = nid
            ys, xs = np.nonzero(part)
            comps[nid] = {"id": nid, "x": int(x0 + xs.min()), "y": int(y0 + ys.min()),
                          "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1),
                          "area": int(part.sum()), "cx": float(x0 + xs.mean()), "cy": float(y0 + ys.mean()),
                          "kind": "pending"}


def analyze(bgr, alpha, cfg, log=None):
    H, W = bgr.shape[:2]
    mask, core, bg, dist = foreground_mask(bgr, alpha, cfg)
    if alpha is None:
        labels, text_px = detect_text(bgr, mask, dist, W, H)
    else:
        labels, text_px = [], np.zeros((H, W), bool)
    mask = mask & ~text_px
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_px = max(6, int(1.2e-5 * W * H))
    comps = {}
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_px:
            continue
        comps[i] = {
            "id": i, "x": int(stats[i, cv2.CC_STAT_LEFT]), "y": int(stats[i, cv2.CC_STAT_TOP]),
            "w": int(stats[i, cv2.CC_STAT_WIDTH]), "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            "area": area, "cx": float(cent[i][0]), "cy": float(cent[i][1]), "kind": "pending",
        }
    for lb in labels:
        bx0, by0, bx1, by1 = lb["box"]
        for cid in list(comps.keys()):
            c = comps[cid]
            if c["x"] >= bx0 - 4 and c["y"] >= by0 - 4 and c["x"] + c["w"] <= bx1 + 4 and c["y"] + c["h"] <= by1 + 4:
                lb["px"] = lb["px"] | (lab == cid)
                del comps[cid]
    tol = max(cfg.tolerance, 0.1)
    if alpha is None:
        for thr, factor in ((42, 1.5), (60, 2.2), (80, 2.2), (100, 2.2)):
            split_merged(lab, comps, dist >= thr * tol, mask, H, W, factor)
    else:
        split_merged(lab, comps, core, mask, H, W, 1.5)
    split_necks(lab, comps, H)

    banners = []
    for c in comps.values():
        if c["kind"] != "pending" or c["area"] < 0.0015 * W * H:
            continue
        aspect = c["w"] / max(1, c["h"])
        if not 1.7 <= aspect <= 9.0:
            continue
        lr, ud = symmetry(lab, c)
        if lr >= 0.88 and ud >= 0.80 and c["area"] / (c["w"] * c["h"]) >= 0.5:
            c["kind"] = "banner"
            banners.append({"kind": "banner", "box": box_of(c), "members": [c["id"]],
                            "cx": c["cx"], "cy": c["cy"], "text": "", "area": c["area"]})

    pool = [c for c in comps.values() if c["kind"] == "pending"]
    frames, strays, ignored = [], [], []
    if pool:
        significant = [c["area"] for c in pool if c["h"] >= 0.07 * H or c["area"] >= 0.0015 * W * H]
        ref = float(np.median(significant)) if significant else float(max(c["area"] for c in pool))
        bodies = [c for c in pool if c["area"] >= 0.2 * ref or c["h"] >= 0.09 * H]
        body_ids = {c["id"] for c in bodies}
        fragments = sorted([c for c in pool if c["id"] not in body_ids], key=lambda c: -c["area"])
        gimg = np.zeros((H, W), np.int32)
        member_of = {}
        for gi, c in enumerate(bodies):
            x0, y0, x1, y1 = box_of(c)
            sub = lab[y0:y1, x0:x1] == c["id"]
            gimg[y0:y1, x0:x1][sub] = gi + 1
            member_of[c["id"]] = gi
        med_h = float(np.median([c["h"] for c in bodies])) if bodies else 0.0
        reach = max(6.0, cfg.attach * med_h)
        pending = list(fragments)
        for _ in range(4):
            if not pending or not bodies:
                break
            left = []
            progressed = False
            for f in pending:
                pad = int(math.ceil(reach)) + 2
                x0, y0 = max(0, f["x"] - pad), max(0, f["y"] - pad)
                x1, y1 = min(W, f["x"] + f["w"] + pad), min(H, f["y"] + f["h"] + pad)
                fm = lab[y0:y1, x0:x1] == f["id"]
                dt = cv2.distanceTransform((~fm).astype(np.uint8), cv2.DIST_L2, 3)
                win = gimg[y0:y1, x0:x1]
                sel = win > 0
                best_gi, best_d = None, reach
                if sel.any():
                    ids = win[sel]
                    dv = dt[sel]
                    for gid in np.unique(ids):
                        dmin = float(dv[ids == gid].min())
                        if dmin <= best_d:
                            best_d, best_gi = dmin, int(gid) - 1
                if best_gi is None:
                    left.append(f)
                    continue
                member_of[f["id"]] = best_gi
                gimg[y0:y1, x0:x1][fm] = best_gi + 1
                progressed = True
            pending = left
            if not progressed:
                break

        groups = [{"members": [], "boxes": []} for _ in bodies]
        for cid, gi in member_of.items():
            groups[gi]["members"].append(cid)
            groups[gi]["boxes"].append(box_of(comps[cid]))
        merged = {gi: groups[gi]["members"] for gi in range(len(groups))}
        for members in merged.values():
            cs = [comps[m] for m in members]
            area = sum(c["area"] for c in cs)
            frames.append({
                "kind": "frame", "members": members, "box": union_box([box_of(c) for c in cs]),
                "area": area, "cx": sum(c["cx"] * c["area"] for c in cs) / area,
                "cy": sum(c["cy"] * c["area"] for c in cs) / area,
                "section": "", "index": 0, "text": "",
            })
        for f in pending:
            corner = f["cx"] > 0.82 * W and f["cy"] > 0.82 * H
            kind = "stray"
            if f["area"] < 0.03 * ref:
                kind = "noise"
            elif corner and f["area"] < 0.004 * W * H:
                kind = "watermark"
            item = {"kind": kind,
                    "box": box_of(f), "members": [f["id"]], "cx": f["cx"], "cy": f["cy"],
                    "area": f["area"], "text": ""}
            (strays if kind == "stray" else ignored).append(item)

    owner = np.zeros((H, W), np.int32)
    items = []
    next_id = 1
    for group in (frames, labels, banners, strays, ignored):
        for it in group:
            it["id"] = next_id
            next_id += 1
            items.append(it)
            if it["kind"] == "label":
                owner[it["px"] & (owner == 0)] = it["id"]
                continue
            for m in it["members"]:
                c = comps[m]
                x0, y0, x1, y1 = box_of(c)
                sub = lab[y0:y1, x0:x1] == m
                owner[y0:y1, x0:x1][sub] = it["id"]
    return {"W": W, "H": H, "bg": bg, "owner": owner, "frames": frames, "labels": labels,
            "banners": banners, "strays": strays, "ignored": ignored}


def order_rows(items, tol):
    items = sorted(items, key=lambda i: i["cy"])
    rows = []
    for it in items:
        if rows and abs(it["cy"] - rows[-1]["cy"]) <= tol:
            rows[-1]["items"].append(it)
            n = len(rows[-1]["items"])
            rows[-1]["cy"] += (it["cy"] - rows[-1]["cy"]) / n
        else:
            rows.append({"cy": it["cy"], "items": [it]})
    out = []
    for r in rows:
        out.extend(sorted(r["items"], key=lambda i: i["cx"]))
    return out


def make_ocr(mode):
    if mode == "off":
        return None
    order = ["rapid", "easy", "tess"] if mode == "auto" else [mode]
    for kind in order:
        try:
            if kind == "rapid":
                try:
                    from rapidocr_onnxruntime import RapidOCR
                    engine = RapidOCR()

                    def run(img):
                        res, _ = engine(img)
                        return " ".join(r[1] for r in res) if res else ""
                    return run
                except ImportError:
                    from rapidocr import RapidOCR
                    engine = RapidOCR()

                    def run(img):
                        res = engine(img)
                        return " ".join(res.txts) if res is not None and res.txts else ""
                    return run
            if kind == "easy":
                import easyocr
                reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                return lambda img: " ".join(reader.readtext(img, detail=0))
            if kind == "tess":
                import pytesseract
                pytesseract.get_tesseract_version()
                return lambda img: pytesseract.image_to_string(img, config="--psm 7")
        except Exception:
            continue
    return None


def read_label_text(ocr, bgr, box):
    x0, y0, x1, y1 = box
    p = 6
    H, W = bgr.shape[:2]
    crop = bgr[max(0, y0 - p):min(H, y1 + p), max(0, x0 - p):min(W, x1 + p)]
    gray = 255 - crop.min(axis=2)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    try:
        return ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    except Exception:
        return ""


def read_banner_text(ocr, bgr, alpha, box):
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1].copy()
    if alpha is not None:
        crop = (crop.astype(np.float32) * (alpha[y0:y1, x0:x1, None] / 255.0)).astype(np.uint8)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try:
        return ocr(crop)
    except Exception:
        return ""


def name_labels(layout, bgr, alpha, cfg, ocr, stem, log):
    labels = order_rows(layout["labels"], 1.5 * (np.median([l["box"][3] - l["box"][1] for l in layout["labels"]]) if layout["labels"] else 20))
    given = [slugify(s) for s in cfg.labels]
    used = {}
    for i, lb in enumerate(labels):
        text = ""
        if i < len(given):
            text = given[i]
        elif ocr is not None:
            text = slugify(read_label_text(ocr, bgr, lb["box"]), "")
        if not text:
            text = f"{slugify(stem)}_section_{i + 1:02d}"
        used[text] = used.get(text, 0) + 1
        if used[text] > 1:
            text = f"{text}_{used[text]}"
        lb["text"] = text
    layout["labels"] = labels
    if given and len(given) != len(labels):
        say(log, f"  note: {len(given)} names given but {len(labels)} labels detected")
    return labels


def assign_sections(layout, stem):
    labels = layout["labels"]
    frames = layout["frames"]
    if not labels:
        for f in frames:
            f["section"] = "frames"
        return
    lh = float(np.median([l["box"][3] - l["box"][1] for l in labels]))
    xtol = 0.01 * layout["W"]
    for f in frames:
        ref_y = f["box"][1] + 0.75 * (f["box"][3] - f["box"][1])
        above = [l for l in labels if l["cy"] < ref_y]
        if not above:
            near = min(labels, key=lambda l: math.hypot(l["cx"] - f["cx"], l["cy"] - f["cy"]))
            f["section"] = near["text"]
            continue
        best_cy = max(l["cy"] for l in above)
        band = [l for l in above if abs(l["cy"] - best_cy) <= 1.5 * lh]
        left_ok = [l for l in band if l["box"][0] <= f["cx"] + xtol]
        if left_ok:
            pick = max(left_ok, key=lambda l: l["box"][0])
        else:
            pick = min(band, key=lambda l: abs(l["box"][0] - f["cx"]))
        f["section"] = pick["text"]


def build_tasks(layout, bgr, alpha, cfg, items, tag):
    H, W = bgr.shape[:2]
    tasks = []
    owner = layout["owner"]
    for it in items:
        x0, y0, x1, y1 = it["box"]
        pad = max(12, int(0.18 * max(x1 - x0, y1 - y0)))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
        own = (owner[cy0:cy1, cx0:cx1] == it["id"]).astype(np.uint8)
        other = ((owner[cy0:cy1, cx0:cx1] > 0) & (owner[cy0:cy1, cx0:cx1] != it["id"])).astype(np.uint8)
        crop = bgr[cy0:cy1, cx0:cx1].copy()
        acrop = alpha[cy0:cy1, cx0:cx1].copy() if alpha is not None else None
        pl, pt = pad - (x0 - cx0), pad - (y0 - cy0)
        pr, pb = pad - (cx1 - x1), pad - (cy1 - y1)
        pl, pt, pr, pb = max(0, pl), max(0, pt), max(0, pr), max(0, pb)
        if pl or pt or pr or pb:
            bgv = tuple(int(v) for v in layout["bg"]) if layout["bg"] is not None else (0, 0, 0)
            crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=bgv)
            own = cv2.copyMakeBorder(own, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
            other = cv2.copyMakeBorder(other, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
            if acrop is not None:
                acrop = cv2.copyMakeBorder(acrop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
        tasks.append({"item": it["id"], "tag": tag, "bgr": crop, "alpha": acrop, "own": own,
                      "other": other, "bg": layout["bg"], "cfg": cfg,
                      "grow": max(4, int(0.06 * max(x1 - x0, y1 - y0)))})
    return tasks


def render_item(task):
    cfg = task["cfg"]
    crop, acrop, own, other = task["bgr"], task["alpha"], task["own"], task["other"]
    bg = task["bg"]
    h0, w0 = crop.shape[:2]
    scale = fit_scale(h0, w0, cfg.scale, cfg.max_side)
    size = (w0 * scale, h0 * scale)
    allowed_small = cv2.dilate(own, ellipse(task["grow"])) > 0
    allowed_small &= other == 0
    allowed = cv2.resize(allowed_small.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    allowed = cv2.GaussianBlur(allowed, (0, 0), max(1.0, scale * 0.5))
    other_up = cv2.resize(other, size, interpolation=cv2.INTER_NEAREST) > 0

    if acrop is not None:
        a = acrop.astype(np.float32) / 255.0
        a[other > 0] = 0.0
        pre = crop.astype(np.float32) * a[..., None]
        pre_up = upscale_hq(pre, scale, cfg.refine)
        a_up = np.clip(upscale_hq(a, scale, cfg.refine), 0.0, 1.0)
        color = np.clip(pre_up / np.maximum(a_up[..., None], 1e-3), 0, 255)
        color = np.where(a_up[..., None] > 0.02, color, 0)
        alpha_out = a_up
        solid = a_up > 0.5
        if solid.any():
            color = propagate_colors(color.astype(np.float32), solid)
        color = sharpen_layer(color, cfg.sharpen, scale)
    else:
        work = crop.copy()
        work[other > 0] = np.clip(bg, 0, 255).astype(np.uint8)
        if cfg.denoise > 0:
            work = cv2.fastNlMeansDenoisingColored(work, None, cfg.denoise, cfg.denoise, 5, 15)
        big = upscale_hq(work, scale, cfg.refine)
        big_u8 = np.clip(big + 0.5, 0, 255).astype(np.uint8)
        alpha, fg_est = chroma_alpha(big_u8, bg, cfg.tolerance, scale)
        alpha = np.clip((alpha - 0.04) / 0.92, 0.0, 1.0)
        alpha = alpha * allowed
        alpha[other_up] = 0.0
        alpha = np.clip(alpha, 0.0, 1.0)
        layer = estimate_foreground(big, alpha, bg, fg_est)
        layer = despill(layer, alpha, bg, cfg.spill, scale)
        color = sharpen_layer(layer, cfg.sharpen, scale)
        alpha_out = alpha
        if cfg.feather > 0:
            alpha_out = cv2.GaussianBlur(alpha, (0, 0), cfg.feather * scale * 0.5)

    a8 = np.clip(alpha_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    c8 = np.clip(color + 0.5, 0, 255).astype(np.uint8)
    ys, xs = np.where(a8 > 8)
    if len(xs) == 0:
        return {"item": task["item"], "tag": task["tag"], "rgba": None}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([c8, a8])[y0:y1, x0:x1]
    wts = a8[y0:y1, x0:x1].astype(np.float64)
    ax = float((wts.sum(axis=0) * np.arange(wts.shape[1])).sum() / max(wts.sum(), 1.0))
    m = cfg.margin
    rgba = cv2.copyMakeBorder(rgba, m, m, m, m, cv2.BORDER_REPLICATE)
    rgba[:m, :, 3] = 0
    rgba[-m:, :, 3] = 0
    rgba[:, :m, 3] = 0
    rgba[:, -m:, 3] = 0
    return {"item": task["item"], "tag": task["tag"], "rgba": rgba, "ax": ax + m, "scale": scale}


def compose(group, margin):
    ax_max = max(max(r["ax"], r["rgba"].shape[1] - r["ax"]) for r in group)
    cw = int(math.ceil(2 * ax_max))
    ch = max(r["rgba"].shape[0] for r in group)
    for r in group:
        canvas = np.zeros((ch, cw, 4), np.uint8)
        h, w = r["rgba"].shape[:2]
        x0 = int(round(cw / 2.0 - r["ax"]))
        x0 = max(0, min(cw - w, x0))
        canvas[ch - h:ch, x0:x0 + w] = r["rgba"]
        r["rgba"] = canvas
        r["ax"] = cw / 2.0


def checker(w, h, size=12):
    y, x = np.mgrid[0:h, 0:w]
    tile = (((x // size) + (y // size)) % 2) * 40 + 190
    return np.dstack([tile, tile, tile]).astype(np.float32)


def contact_sheet(records, path, row_h=110, max_w=2000):
    by_section = {}
    for r in records:
        by_section.setdefault(r["section"], []).append(r)
    rows = []
    for section, recs in by_section.items():
        tiles = []
        for r in recs:
            img = r["rgba"]
            ratio = row_h / img.shape[0]
            tiles.append(cv2.resize(img, (max(1, int(img.shape[1] * ratio)), row_h), interpolation=cv2.INTER_AREA))
        line, cur = [], []
        cw = 0
        for t in tiles:
            if cw + t.shape[1] > max_w and cur:
                line.append(cur)
                cur, cw = [], 0
            cur.append(t)
            cw += t.shape[1] + 6
        if cur:
            line.append(cur)
        for k, group in enumerate(line):
            rows.append((section if k == 0 else "", group))
    if not rows:
        return
    width = max_w
    strip_h = row_h + 22
    sheet = np.zeros((strip_h * len(rows), width, 3), np.float32)
    sheet[:] = checker(width, strip_h * len(rows))
    for ri, (title, group) in enumerate(rows):
        y = ri * strip_h
        cv2.putText(sheet, title, (4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        x = 4
        for t in group:
            if x + t.shape[1] > width:
                break
            a = t[..., 3:4].astype(np.float32) / 255.0
            region = sheet[y + 20:y + 20 + row_h, x:x + t.shape[1]]
            region[:] = t[..., :3] * a + region * (1 - a)
            x += t.shape[1] + 6
    write_png(path, np.dstack([np.clip(sheet, 0, 255).astype(np.uint8), np.full(sheet.shape[:2], 255, np.uint8)]))


def draw_debug(bgr, layout, path):
    img = bgr.copy()
    colors = {"frame": (255, 0, 255), "label": (0, 255, 255), "banner": (0, 140, 255),
              "stray": (0, 0, 255), "watermark": (160, 160, 160), "noise": (120, 120, 120)}
    for group in ("frames", "labels", "banners", "strays", "ignored"):
        for it in layout[group]:
            x0, y0, x1, y1 = it["box"]
            col = colors[it["kind"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            tag = f"{it.get('section') or it.get('text') or it['kind']}"
            if it["kind"] == "frame":
                tag = f"{it['section']}:{it['index']}"
            cv2.putText(img, tag[:28], (x0 + 2, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, tag[:28], (x0 + 2, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    write_png(path, np.dstack([img, np.full(img.shape[:2], 255, np.uint8)]))


def collect_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for n in sorted(os.listdir(p)):
                if n.lower().endswith(IMAGE_EXTENSIONS):
                    files.append(os.path.join(p, n))
        elif os.path.isfile(p):
            files.append(p)
    return files


def run_pipeline(paths, out_dir, cfg, log=None, progress=None, cancel=None):
    files = collect_inputs(paths)
    if not files:
        say(log, "No images found.")
        return 1
    os.makedirs(out_dir, exist_ok=True)
    ocr = make_ocr(cfg.ocr)
    say(log, "OCR: " + ("enabled" if ocr else "not available, using generic section names"))

    sheets = []
    tasks = []
    for path in files:
        if cancel is not None and cancel.is_set():
            return 2
        stem = os.path.splitext(os.path.basename(path))[0]
        say(log, f"Analyzing {os.path.basename(path)}")
        bgr, alpha = read_image(path)
        layout = analyze(bgr, alpha, cfg, log)
        name_labels(layout, bgr, alpha, cfg, ocr, stem, log)
        assign_sections(layout, stem)

        char = slugify(cfg.name) if cfg.name else ""
        if not char and ocr is not None and layout["banners"]:
            votes = {}
            for b in layout["banners"]:
                t = slugify(read_banner_text(ocr, bgr, alpha, b["box"]), "")
                if t:
                    votes[t] = votes.get(t, 0) + 1
            if votes:
                char = max(votes, key=votes.get)
        if not char:
            char = slugify(stem)

        by_section = {}
        for f in layout["frames"]:
            by_section.setdefault(f["section"], []).append(f)
        ordered = []
        for section, fs in by_section.items():
            mh = float(np.median([f["box"][3] - f["box"][1] for f in fs]))
            for k, f in enumerate(order_rows(fs, 0.5 * mh), 1):
                f["index"] = k
                ordered.append(f)
        layout["frames"] = ordered

        say(log, f"  character '{char}': {len(layout['frames'])} frames, {len(layout['labels'])} labels, "
                 f"{len(layout['banners'])} banners, {len(layout['strays'])} review items")
        for lb in layout["labels"]:
            n = sum(1 for f in layout["frames"] if f["section"] == lb["text"])
            say(log, f"    {lb['text']}: {n}")

        sheet = {"path": path, "stem": stem, "char": char, "layout": layout, "bgr": bgr, "alpha": alpha}
        sheets.append(sheet)
        sources = [(layout["frames"], "frame")]
        if layout["banners"]:
            sources.append((layout["banners"], "banner"))
        if cfg.review and layout["strays"]:
            sources.append((layout["strays"], "stray"))
        for items, tag in sources:
            for t in build_tasks(layout, bgr, alpha, cfg, items, tag):
                t["sheet"] = len(sheets) - 1
                tasks.append(t)
        if cfg.debug:
            draw_debug(bgr, layout, os.path.join(out_dir, "_debug", f"{stem}_detect.png"))

    workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    results = []
    total = len(tasks)
    say(log, f"Rendering {total} items at x{cfg.scale} with {workers} worker(s)")
    if workers == 1 or total <= 1:
        for i, t in enumerate(tasks, 1):
            if cancel is not None and cancel.is_set():
                return 2
            r = render_item(t)
            r["sheet"] = t["sheet"]
            results.append(r)
            if progress:
                progress(i, total)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, r in enumerate(pool.map(render_item, tasks, chunksize=1), 1):
                r["sheet"] = tasks[i - 1]["sheet"]
                results.append(r)
                if progress:
                    progress(i, total)
                if cancel is not None and cancel.is_set():
                    return 2

    lookup = {}
    for r in results:
        lookup[(r["sheet"], r["tag"], r["item"])] = r

    per_char = {}
    counters = {}
    for si, sheet in enumerate(sheets):
        layout = sheet["layout"]
        char = sheet["char"]
        recs = per_char.setdefault(char, {"frames": [], "banners": [], "strays": []})
        for f in layout["frames"]:
            r = lookup.get((si, "frame", f["id"]))
            if r is None or r["rgba"] is None:
                continue
            key = (char, f["section"])
            counters[key] = counters.get(key, 0) + 1
            r.update({"section": f["section"], "number": counters[key], "box": f["box"], "sheet_name": sheet["stem"]})
            recs["frames"].append(r)
        for kind, tag in (("banners", "banner"), ("strays", "stray")):
            for k, it in enumerate(layout[kind], 1):
                r = lookup.get((si, tag, it["id"]))
                if r is None or r["rgba"] is None:
                    continue
                r.update({"section": tag, "number": k, "box": it["box"], "sheet_name": sheet["stem"]})
                recs[kind].append(r)

    for char, recs in per_char.items():
        frames = recs["frames"]
        if cfg.canvas in ("section", "character") and frames:
            if cfg.canvas == "character":
                compose(frames, cfg.margin)
            else:
                groups = {}
                for r in frames:
                    groups.setdefault(r["section"], []).append(r)
                for g in groups.values():
                    compose(g, cfg.margin)
        char_dir = os.path.join(out_dir, char)
        manifest = []
        for r in frames:
            rel = os.path.join(r["section"], f"{char}_{r['section']}_{r['number']:02d}.png")
            write_png(os.path.join(char_dir, rel), r["rgba"])
            manifest.append({"file": rel.replace("\\", "/"), "section": r["section"], "index": r["number"],
                             "sheet": r["sheet_name"], "sheet_box": list(r["box"]),
                             "width": int(r["rgba"].shape[1]), "height": int(r["rgba"].shape[0]),
                             "anchor_x": round(float(r["ax"]), 2), "anchor_y": int(r["rgba"].shape[0] - cfg.margin),
                             "scale": r["scale"]})
        for r in recs["banners"]:
            rel = os.path.join("banners", f"{char}_banner_{r['number']:02d}.png")
            write_png(os.path.join(char_dir, rel), r["rgba"])
        for r in recs["strays"]:
            rel = os.path.join("_review", f"{r['sheet_name']}_stray_{r['number']:02d}.png")
            write_png(os.path.join(char_dir, rel), r["rgba"])
        if manifest:
            mpath = os.path.join(char_dir, "manifest.json")
            sheet_names = {m["sheet"] for m in manifest}
            old = []
            if os.path.exists(mpath):
                try:
                    with open(mpath, "r", encoding="utf-8") as fh:
                        old = [m for m in json.load(fh).get("frames", []) if m.get("sheet") not in sheet_names]
                except Exception:
                    old = []
            with open(mpath, "w", encoding="utf-8") as fh:
                json.dump({"character": char, "frames": old + manifest}, fh, indent=2)
        if cfg.debug and frames:
            contact_sheet(frames, os.path.join(char_dir, "contact_sheet.png"))
        say(log, f"Saved {len(frames)} frames -> {char_dir}")
    say(log, "Done.")
    return 0


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    class App:
        def __init__(self, root):
            self.root = root
            root.title("LRD-TECH Sprite Slicer Pro")
            root.geometry("820x760")
            self.files = []
            self.out = tk.StringVar(value=os.path.join(os.getcwd(), "sliced"))
            self.name = tk.StringVar()
            self.scale = tk.StringVar(value="4")
            self.canvas = tk.StringVar(value="tight")
            self.ocr = tk.StringVar(value="auto")
            self.labels = tk.StringVar()
            self.debug = tk.BooleanVar(value=True)
            self.events = queue.Queue()
            self.cancel_flag = threading.Event()
            self.build()

        def build(self):
            r = self.root
            tk.Label(r, text="LRD-TECH Sprite Slicer Pro", font=("Helvetica", 16, "bold")).pack(pady=8)
            top = tk.Frame(r)
            top.pack(pady=4)
            tk.Button(top, text="Add Sheets", command=self.add_files).grid(row=0, column=0, padx=6)
            tk.Button(top, text="Add Folder", command=self.add_folder).grid(row=0, column=1, padx=6)
            tk.Button(top, text="Clear", command=self.clear).grid(row=0, column=2, padx=6)
            tk.Button(top, text="Output Folder", command=self.pick_out).grid(row=0, column=3, padx=6)
            self.box = tk.Listbox(r, height=6)
            self.box.pack(fill="x", padx=14, pady=4)
            tk.Label(r, textvariable=self.out, anchor="w").pack(fill="x", padx=14)

            opts = tk.LabelFrame(r, text="Options")
            opts.pack(fill="x", padx=14, pady=6)
            tk.Label(opts, text="Character name").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            tk.Entry(opts, textvariable=self.name, width=22).grid(row=0, column=1, padx=6)
            tk.Label(opts, text="Upscale").grid(row=0, column=2, sticky="w", padx=6)
            ttk.Combobox(opts, textvariable=self.scale, values=["1", "2", "3", "4", "6", "8"], width=4,
                         state="readonly").grid(row=0, column=3, padx=6)
            tk.Label(opts, text="Canvas").grid(row=1, column=0, sticky="w", padx=6, pady=3)
            ttk.Combobox(opts, textvariable=self.canvas, values=["tight", "section", "character"], width=12,
                         state="readonly").grid(row=1, column=1, padx=6, sticky="w")
            tk.Label(opts, text="OCR").grid(row=1, column=2, sticky="w", padx=6)
            ttk.Combobox(opts, textvariable=self.ocr, values=["auto", "off", "rapid", "easy", "tess"], width=6,
                         state="readonly").grid(row=1, column=3, padx=6)
            tk.Label(opts, text="Section names (comma list, reading order)").grid(row=2, column=0, columnspan=2, sticky="w", padx=6)
            tk.Entry(opts, textvariable=self.labels, width=60).grid(row=3, column=0, columnspan=4, padx=6, pady=3, sticky="w")
            tk.Checkbutton(opts, text="Write debug overlay and contact sheet", variable=self.debug).grid(
                row=4, column=0, columnspan=3, sticky="w", padx=6)

            self.progress = tk.DoubleVar()
            ttk.Progressbar(r, orient="horizontal", length=560, mode="determinate", variable=self.progress).pack(pady=8)
            btns = tk.Frame(r)
            btns.pack()
            self.start_btn = tk.Button(btns, text="Start", command=self.start)
            self.start_btn.grid(row=0, column=0, padx=8)
            self.cancel_btn = tk.Button(btns, text="Cancel", command=self.cancel, state=tk.DISABLED)
            self.cancel_btn.grid(row=0, column=1, padx=8)
            self.log = ScrolledText(r, height=18)
            self.log.pack(fill="both", expand=True, padx=14, pady=8)

        def add_files(self):
            paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")])
            for p in paths:
                self.files.append(p)
                self.box.insert(tk.END, p)

        def add_folder(self):
            folder = filedialog.askdirectory()
            if folder:
                for p in collect_inputs([folder]):
                    self.files.append(p)
                    self.box.insert(tk.END, p)

        def clear(self):
            self.files = []
            self.box.delete(0, tk.END)

        def pick_out(self):
            folder = filedialog.askdirectory()
            if folder:
                self.out.set(folder)

        def config(self):
            labels = tuple(s.strip() for s in self.labels.get().split(",") if s.strip())
            return Config(scale=int(self.scale.get()), canvas=self.canvas.get(), name=self.name.get().strip(),
                          labels=labels, ocr=self.ocr.get(), debug=bool(self.debug.get()))

        def start(self):
            if not self.files:
                messagebox.showerror("Error", "Add at least one sheet.")
                return
            self.cancel_flag.clear()
            self.progress.set(0)
            self.start_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
            cfg = self.config()
            threading.Thread(target=self.work, args=(list(self.files), self.out.get(), cfg), daemon=True).start()
            self.root.after(100, self.poll)

        def cancel(self):
            self.cancel_flag.set()

        def work(self, files, out, cfg):
            try:
                run_pipeline(files, out, cfg, log=lambda m: self.events.put(("log", m)),
                             progress=lambda i, n: self.events.put(("progress", i * 100.0 / n)),
                             cancel=self.cancel_flag)
            except Exception as exc:
                self.events.put(("log", f"ERROR: {exc}"))
            self.events.put(("done", None))

        def poll(self):
            finished = False
            try:
                while True:
                    kind, val = self.events.get_nowait()
                    if kind == "log":
                        self.log.insert(tk.END, val + "\n")
                        self.log.see(tk.END)
                    elif kind == "progress":
                        self.progress.set(val)
                    elif kind == "done":
                        finished = True
                        self.start_btn.config(state=tk.NORMAL)
                        self.cancel_btn.config(state=tk.DISABLED)
            except queue.Empty:
                pass
            if not finished:
                self.root.after(100, self.poll)

    root = tk.Tk()
    App(root)
    root.mainloop()


def build_parser():
    p = argparse.ArgumentParser(description="LRD-TECH Sprite Slicer Pro")
    p.add_argument("inputs", nargs="*", help="Sheet images or folders")
    p.add_argument("-o", "--output", default="sliced")
    p.add_argument("--name", default="")
    p.add_argument("--scale", type=int, default=4, choices=[1, 2, 3, 4, 6, 8])
    p.add_argument("--canvas", choices=["tight", "section", "character"], default="tight")
    p.add_argument("--labels", default="")
    p.add_argument("--labels-file", default="")
    p.add_argument("--ocr", choices=["auto", "off", "rapid", "easy", "tess"], default="auto")
    p.add_argument("--bg", default="auto")
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--spill", type=float, default=1.0)
    p.add_argument("--sharpen", type=float, default=0.6)
    p.add_argument("--denoise", type=float, default=2.0)
    p.add_argument("--feather", type=float, default=0.5)
    p.add_argument("--refine", type=int, default=2)
    p.add_argument("--margin", type=int, default=8)
    p.add_argument("--attach", type=float, default=0.16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-debug", action="store_true")
    p.add_argument("--no-review", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if not args.inputs:
        run_gui()
        return 0
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if args.labels_file and os.path.exists(args.labels_file):
        with open(args.labels_file, "r", encoding="utf-8") as fh:
            labels = [ln.strip() for ln in fh if ln.strip()]
    cfg = Config(scale=args.scale, tolerance=args.tolerance, spill=args.spill, sharpen=args.sharpen,
                 denoise=args.denoise, feather=args.feather, refine=args.refine, margin=args.margin,
                 canvas=args.canvas, name=args.name, labels=tuple(labels), ocr=args.ocr, bg=args.bg,
                 attach=args.attach, workers=args.workers, debug=not args.no_debug, review=not args.no_review)
    return run_pipeline(args.inputs, args.output, cfg)


if __name__ == "__main__":
    sys.exit(main())
