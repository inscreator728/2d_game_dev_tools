import argparse
import os
import queue
import sys
import threading
from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
FORMAT_EXTENSIONS = {"PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff", "JPG": ".jpg"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_AI_SESSIONS = {}
_AI_STATE = {"checked": False, "available": False}


@dataclass
class Config:
    engine: str = "auto"
    scale: int = 2
    tolerance: float = 1.0
    spill: float = 1.0
    sharpen: float = 0.7
    denoise: float = 3.0
    feather: float = 0.8
    shrink: int = 0
    reach: float = 0.10
    trim: bool = False
    ai_model: str = "isnet-general-use"
    flatten_color: tuple = (255, 255, 255)
    max_side: int = 6000


def say(log, message):
    if log:
        log(message)
    else:
        print(message)


def ellipse(radius):
    radius = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


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
    return np.ascontiguousarray(img[:, :, :3])


def flatten_on_color(bgra, color):
    alpha = bgra[..., 3:4].astype(np.float32) / 255.0
    background = np.array(color[::-1], np.float32)
    out = bgra[..., :3].astype(np.float32) * alpha + background * (1.0 - alpha)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def write_image(path, bgra, fmt, cfg):
    ext = FORMAT_EXTENSIONS[fmt]
    params = []
    image = bgra
    if fmt == "JPG":
        image = flatten_on_color(bgra, cfg.flatten_color)
        params = [cv2.IMWRITE_JPEG_QUALITY, 98]
    elif fmt == "PNG":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif fmt == "WEBP":
        params = [cv2.IMWRITE_WEBP_QUALITY, 101]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    buf.tofile(path)


def estimate_background(bgr):
    h, w = bgr.shape[:2]
    t = max(2, min(h, w) // 50)
    border = np.concatenate([
        bgr[:t].reshape(-1, 3),
        bgr[-t:].reshape(-1, 3),
        bgr[:, :t].reshape(-1, 3),
        bgr[:, -t:].reshape(-1, 3),
    ])
    hsv = cv2.cvtColor(border.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    greenish = (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 120) & (hsv[:, 2] >= 100)
    pool = border[greenish] if greenish.sum() > 50 else border
    return np.median(pool, axis=0).astype(np.float32)


def key_info(bg):
    ch = int(np.argmax(bg))
    others = [i for i in range(3) if i != ch]
    strength = float(bg[ch] - max(bg[others[0]], bg[others[1]]))
    return ch, others[0], others[1], strength


def lab_distance(bgr, bg):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2LAB)
    ref = ref.astype(np.float32).reshape(3)
    dl = (lab[..., 0] - ref[0]) * (100.0 / 255.0)
    da = lab[..., 1] - ref[1]
    db = lab[..., 2] - ref[2]
    return np.sqrt(0.5 * dl * dl + da * da + db * db)


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


def chroma_alpha(bgr, cfg, scale):
    h, w = bgr.shape[:2]
    img = bgr.astype(np.float32)
    bg = estimate_background(bgr)
    d = lab_distance(bgr, bg)
    tol = max(cfg.tolerance, 0.1)
    low, high = 12.0 * tol, 42.0 * tol

    candidate = d < low
    min_area = max(64, int(0.0008 * h * w))
    accepted = accepted_background(candidate, min_area)

    dist_bg = cv2.distanceTransform((~accepted).astype(np.uint8), cv2.DIST_L2, 3)
    band = max(2.0, 2.5 * scale)
    reach = max(band + 1.0, cfg.reach * min(h, w))

    core = (~accepted) & (d >= high) & (dist_bg > band)
    unknown = (~accepted) & (~core) & (dist_bg <= reach)

    alpha = np.ones((h, w), np.float32)
    alpha[accepted] = 0.0

    if not core.any():
        core = (~accepted) & (d >= high)
    if not core.any():
        return alpha, img.copy(), bg, d, accepted, high

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
    return alpha.astype(np.float32), fg_est, bg, d, accepted, high


def ai_pipeline(bgr, ai, scale):
    img = bgr.astype(np.float32)
    bg = estimate_background(bgr)
    band = max(2.0, 2.5 * scale)
    core = cv2.erode((ai > 0.97).astype(np.uint8), ellipse(band)) > 0
    if not core.any():
        core = ai > 0.5
    if not core.any():
        return np.zeros(ai.shape, np.float32), img.copy(), bg
    fg_est = propagate_colors(img, core)
    zone = cv2.dilate(((ai > 0.02) & (ai < 0.98)).astype(np.uint8), ellipse(band)) > 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined = guided_filter(gray, ai.astype(np.float32), max(2, int(round(1.5 * scale))), 1e-3)
    alpha = np.where(zone, refined, ai)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32), fg_est, bg


def hybrid_merge(alpha, ai, d, accepted, high):
    h, w = alpha.shape
    m = min(h, w)
    core_px = max(2, int(0.008 * m))
    core = (cv2.erode((ai > 0.92).astype(np.uint8), ellipse(core_px)) > 0) & (~accepted)
    alpha = np.where(core, 1.0, alpha)
    halo_px = max(6.0, 0.05 * m)
    ai_fg = (ai > 0.1).astype(np.uint8)
    if ai_fg.any():
        dist_ai = cv2.distanceTransform(1 - ai_fg, cv2.DIST_L2, 3)
        kill = (dist_ai > halo_px) & (d < high * 1.8)
        alpha = np.where(kill, 0.0, alpha)
    return alpha.astype(np.float32)


def remove_specks(alpha):
    h, w = alpha.shape
    mask = (alpha > 0.1).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return np.where(mask > 0, alpha, 0.0).astype(np.float32)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    largest = area[1:].max()
    min_speck = max(16, int(0.00003 * h * w))
    touches = (x == 0) | (y == 0) | (x + bw >= w) | (y + bh >= h)
    keep = (area >= min_speck) & ~(touches & (area < 0.05 * largest))
    keep[0] = False
    return (alpha * keep[labels]).astype(np.float32)


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
    for name in ("EDSR", "FSRCNN", "ESPCN"):
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


def upscale(bgr, scale):
    if scale <= 1:
        return bgr
    sr = load_superres(scale)
    if sr is not None:
        try:
            return sr.upsample(bgr)
        except Exception:
            pass
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def ai_available():
    if not _AI_STATE["checked"]:
        _AI_STATE["checked"] = True
        try:
            import rembg
            _AI_STATE["available"] = True
        except Exception:
            _AI_STATE["available"] = False
    return _AI_STATE["available"]


def ai_alpha(bgr, model):
    from PIL import Image
    from rembg import new_session, remove
    session = _AI_SESSIONS.get(model)
    if session is None:
        session = new_session(model)
        _AI_SESSIONS[model] = session
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask = remove(rgb, session=session, only_mask=True, post_process_mask=True)
    return np.asarray(mask.convert("L"), np.float32) / 255.0


def resolve_engine(engine, log):
    if engine == "chroma":
        return "chroma"
    if ai_available():
        return "hybrid" if engine == "auto" else engine
    if engine in ("ai", "hybrid"):
        say(log, "rembg is not installed, falling back to chroma engine")
    return "chroma"


def process(bgr, cfg, log=None):
    bgr = np.ascontiguousarray(bgr[:, :, :3])
    h0, w0 = bgr.shape[:2]
    engine = resolve_engine(cfg.engine, log)

    work = bgr
    if cfg.denoise > 0:
        work = cv2.fastNlMeansDenoisingColored(bgr, None, cfg.denoise, cfg.denoise, 5, 15)

    ai_small = None
    if engine in ("ai", "hybrid"):
        try:
            ai_small = ai_alpha(work, cfg.ai_model)
        except Exception as exc:
            say(log, f"AI mask failed ({exc}), using chroma engine")
            engine = "chroma"

    scale = max(1, int(cfg.scale))
    while scale > 1 and max(h0, w0) * scale > cfg.max_side:
        scale -= 1

    big = upscale(work, scale)
    h, w = big.shape[:2]
    scale_f = w / float(w0)
    img = big.astype(np.float32)

    ai = None
    if ai_small is not None:
        ai = np.clip(cv2.resize(ai_small, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)

    if engine == "ai":
        alpha, fg_est, bg = ai_pipeline(big, ai, scale_f)
    else:
        alpha, fg_est, bg, d, accepted, high = chroma_alpha(big, cfg, scale_f)
        if engine == "hybrid":
            alpha = hybrid_merge(alpha, ai, d, accepted, high)

    alpha = np.clip((alpha - 0.04) / 0.92, 0.0, 1.0)
    alpha = remove_specks(alpha)

    alpha_out = alpha
    if cfg.shrink > 0:
        iterations = max(1, int(round(cfg.shrink * max(1.0, scale_f / 2.0))))
        alpha_out = cv2.erode(alpha_out, np.ones((3, 3), np.uint8), iterations=iterations)
    if cfg.feather > 0:
        alpha_out = cv2.GaussianBlur(alpha_out, (0, 0), cfg.feather * scale_f * 0.5)

    layer = estimate_foreground(img, alpha, bg, fg_est)
    layer = despill(layer, alpha, bg, cfg.spill, scale_f)
    layer = sharpen_layer(layer, cfg.sharpen, scale_f)

    color = np.clip(layer + 0.5, 0, 255).astype(np.uint8)
    a8 = np.clip(alpha_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    bgra = np.dstack([color, a8])

    if cfg.trim:
        ys, xs = np.where(a8 > 8)
        if len(xs):
            pad = 4
            y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
            bgra = bgra[y0:y1, x0:x1]
    return bgra


def process_file(src, out_dir, cfg, fmt, log=None):
    bgr = read_image(src)
    bgra = process(bgr, cfg, log)
    stem = os.path.splitext(os.path.basename(src))[0]
    path = os.path.join(out_dir, f"{stem}_nobg{FORMAT_EXTENSIONS[fmt]}")
    write_image(path, bgra, fmt, cfg)
    return bgra, path


def save_pdf(images, path, cfg):
    from PIL import Image
    pages = []
    for bgra in images:
        flat = flatten_on_color(bgra, cfg.flatten_color)
        pages.append(Image.fromarray(cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)))
    if pages:
        pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=150.0)


def collect_images(folder):
    names = sorted(os.listdir(folder))
    return [
        os.path.join(folder, n) for n in names
        if n.lower().endswith(IMAGE_EXTENSIONS) and "_nobg" not in n.lower()
    ]


def checkerboard(w, h, size=12):
    y, x = np.mgrid[0:h, 0:w]
    tile = (((x // size) + (y // size)) % 2) * 40 + 190
    return np.dstack([tile, tile, tile]).astype(np.float32)


def preview_array(bgra, max_w=500, max_h=300):
    h, w = bgra.shape[:2]
    ratio = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
    small = cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)
    alpha = small[..., 3:4].astype(np.float32) / 255.0
    rgb = cv2.cvtColor(small[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    board = checkerboard(nw, nh)
    return np.clip(rgb * alpha + board * (1.0 - alpha), 0, 255).astype(np.uint8)


def run_cli(args):
    cfg = Config(
        engine=args.engine, scale=args.scale, tolerance=args.tolerance, spill=args.spill,
        sharpen=args.sharpen, denoise=args.denoise, feather=args.feather, shrink=args.shrink,
        reach=args.reach, trim=args.trim, ai_model=args.model,
        flatten_color=tuple(int(v) for v in args.flatten.split(",")),
    )
    fmt = args.format.upper()
    if args.input:
        files = [args.input]
    else:
        if not os.path.isdir(args.directory):
            print(f"Error: Directory {args.directory} not found.")
            return 1
        files = collect_images(args.directory)
    if not files:
        print("No images found.")
        return 1

    out = args.output
    if args.input and os.path.splitext(out)[1].lower() in IMAGE_EXTENSIONS:
        out_dir = os.path.dirname(os.path.abspath(out))
    else:
        out_dir = out
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, src in enumerate(files, 1):
        try:
            bgra, path = process_file(src, out_dir, cfg, fmt)
            results.append(bgra)
            print(f"[{i}/{len(files)}] {os.path.basename(src)} -> {path}")
        except Exception as exc:
            print(f"[{i}/{len(files)}] FAILED {os.path.basename(src)}: {exc}")

    if args.pdf and results:
        pdf_path = os.path.join(out_dir, "Results.pdf")
        save_pdf(results, pdf_path, cfg)
        print(f"PDF saved: {pdf_path}")
    print("Done.")
    return 0


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk

    class App:
        def __init__(self, root):
            self.root = root
            root.title("LRD-TECH Background Remover Pro")
            root.geometry("900x860")
            self.files = []
            self.output_folder = ""
            self.history = []
            self.events = queue.Queue()
            self.cancel_flag = threading.Event()
            self.worker = None

            self.engine = tk.StringVar(value="auto")
            self.scale = tk.StringVar(value="2")
            self.fmt = tk.StringVar(value="PNG")
            self.tolerance = tk.DoubleVar(value=1.0)
            self.spill = tk.DoubleVar(value=1.0)
            self.sharpen = tk.DoubleVar(value=0.7)
            self.feather = tk.DoubleVar(value=0.8)
            self.shrink = tk.IntVar(value=0)
            self.trim = tk.BooleanVar(value=False)
            self.pdf = tk.BooleanVar(value=False)
            self.build()

        def build(self):
            r = self.root
            tk.Label(r, text="LRD-TECH Background Remover Pro", font=("Helvetica", 16, "bold")).pack(pady=8)

            top = tk.Frame(r)
            top.pack(pady=4)
            tk.Button(top, text="Select File(s)", command=self.pick_files).grid(row=0, column=0, padx=8)
            tk.Button(top, text="Select Folder", command=self.pick_folder).grid(row=0, column=1, padx=8)
            tk.Button(top, text="Select Save Location", command=self.pick_output).grid(row=0, column=2, padx=8)

            opts = tk.LabelFrame(r, text="Engine and Quality")
            opts.pack(padx=14, pady=6, fill="x")
            tk.Label(opts, text="Engine").grid(row=0, column=0, padx=6, pady=4, sticky="w")
            ttk.Combobox(opts, textvariable=self.engine, values=["auto", "chroma", "hybrid", "ai"],
                         width=10, state="readonly").grid(row=0, column=1, padx=6)
            tk.Label(opts, text="Upscale").grid(row=0, column=2, padx=6, sticky="w")
            ttk.Combobox(opts, textvariable=self.scale, values=["1", "2", "3", "4"],
                         width=4, state="readonly").grid(row=0, column=3, padx=6)
            tk.Label(opts, text="Save as").grid(row=0, column=4, padx=6, sticky="w")
            for i, f in enumerate(["PNG", "WEBP", "TIFF", "JPG"]):
                tk.Radiobutton(opts, text=f, variable=self.fmt, value=f).grid(row=0, column=5 + i, padx=2)

            sliders = tk.Frame(opts)
            sliders.grid(row=1, column=0, columnspan=9, sticky="ew", pady=4)
            self.slider(sliders, "Key tolerance", self.tolerance, 0.4, 2.5, 0.05, 0, 0)
            self.slider(sliders, "Spill removal", self.spill, 0.0, 1.5, 0.05, 0, 1)
            self.slider(sliders, "Sharpen", self.sharpen, 0.0, 1.5, 0.05, 1, 0)
            self.slider(sliders, "Edge feather", self.feather, 0.0, 3.0, 0.1, 1, 1)
            self.slider(sliders, "Edge shrink (px)", self.shrink, 0, 4, 1, 2, 0)
            tk.Checkbutton(sliders, text="Trim to subject", variable=self.trim).grid(row=2, column=1, sticky="w", padx=10)
            tk.Checkbutton(r, text="Also export all results as one PDF", variable=self.pdf).pack()

            self.progress_var = tk.DoubleVar()
            ttk.Progressbar(r, orient="horizontal", length=560, mode="determinate",
                            variable=self.progress_var).pack(pady=10)

            btns = tk.Frame(r)
            btns.pack(pady=4)
            self.start_btn = tk.Button(btns, text="Start Processing", command=self.start, state=tk.DISABLED)
            self.start_btn.grid(row=0, column=0, padx=8)
            self.cancel_btn = tk.Button(btns, text="Cancel", command=self.cancel, state=tk.DISABLED)
            self.cancel_btn.grid(row=0, column=1, padx=8)
            tk.Button(btns, text="View History", command=self.view_history).grid(row=0, column=2, padx=8)

            self.canvas = tk.Canvas(r, width=500, height=300, bg="gray")
            self.canvas.pack(pady=8)
            self.status = tk.Label(r, text="Status: Waiting for input...")
            self.status.pack(pady=6)

        def slider(self, parent, label, var, lo, hi, step, row, col):
            box = tk.Frame(parent)
            box.grid(row=row, column=col, padx=10, pady=2, sticky="w")
            tk.Label(box, text=label, width=16, anchor="w").pack(side="left")
            tk.Scale(box, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                     length=240).pack(side="left")

        def set_status(self, text):
            self.status.config(text=f"Status: {text}")

        def pick_files(self):
            paths = filedialog.askopenfilenames(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
            if paths:
                self.files = list(paths)
                self.set_status(f"{len(self.files)} images selected.")
                self.start_btn.config(state=tk.NORMAL)

        def pick_folder(self):
            folder = filedialog.askdirectory()
            if folder:
                self.files = collect_images(folder)
                self.set_status(f"{len(self.files)} images found in folder.")
                self.start_btn.config(state=tk.NORMAL if self.files else tk.DISABLED)

        def pick_output(self):
            folder = filedialog.askdirectory()
            if folder:
                self.output_folder = folder
                self.set_status(f"Save location: {folder}")

        def config(self):
            return Config(
                engine=self.engine.get(), scale=int(self.scale.get()),
                tolerance=float(self.tolerance.get()), spill=float(self.spill.get()),
                sharpen=float(self.sharpen.get()), feather=float(self.feather.get()),
                shrink=int(self.shrink.get()), trim=bool(self.trim.get()),
            )

        def start(self):
            if not self.output_folder:
                messagebox.showerror("Error", "Please select a save location.")
                return
            if not self.files:
                return
            self.cancel_flag.clear()
            self.progress_var.set(0)
            self.start_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
            cfg, fmt = self.config(), self.fmt.get()
            self.worker = threading.Thread(
                target=self.run_batch, args=(list(self.files), cfg, fmt, self.pdf.get()), daemon=True)
            self.worker.start()
            self.root.after(100, self.poll)

        def cancel(self):
            self.cancel_flag.set()
            self.set_status("Cancelling...")

        def run_batch(self, files, cfg, fmt, want_pdf):
            collected = []
            total = len(files)
            for i, src in enumerate(files, 1):
                if self.cancel_flag.is_set():
                    break
                try:
                    bgra, path = process_file(src, self.output_folder, cfg, fmt,
                                              log=lambda m: self.events.put(("note", m)))
                    collected.append(bgra)
                    self.events.put(("ok", i, total, os.path.basename(src), path, preview_array(bgra)))
                except Exception as exc:
                    self.events.put(("fail", i, total, os.path.basename(src), str(exc)))
            if want_pdf and collected:
                try:
                    pdf_path = os.path.join(self.output_folder, "Results.pdf")
                    save_pdf(collected, pdf_path, cfg)
                    self.events.put(("pdf", pdf_path))
                except Exception as exc:
                    self.events.put(("note", f"PDF failed: {exc}"))
            self.events.put(("done", self.cancel_flag.is_set()))

        def poll(self):
            finished = False
            try:
                while True:
                    ev = self.events.get_nowait()
                    kind = ev[0]
                    if kind == "ok":
                        _, i, total, name, path, prev = ev
                        self.history.append(path)
                        self.progress_var.set(i * 100.0 / total)
                        self.set_status(f"Processed {i}/{total}: {name}")
                        photo = ImageTk.PhotoImage(Image.fromarray(prev))
                        self.canvas.delete("all")
                        self.canvas.create_image(250, 150, image=photo)
                        self.canvas.image = photo
                    elif kind == "fail":
                        _, i, total, name, msg = ev
                        self.progress_var.set(i * 100.0 / total)
                        self.set_status(f"Failed {name}: {msg}")
                    elif kind == "pdf":
                        self.history.append(ev[1])
                    elif kind == "note":
                        self.set_status(ev[1])
                    elif kind == "done":
                        finished = True
                        self.start_btn.config(state=tk.NORMAL)
                        self.cancel_btn.config(state=tk.DISABLED)
                        if ev[1]:
                            self.set_status("Cancelled.")
                        else:
                            messagebox.showinfo("Success", "All images processed.")
            except queue.Empty:
                pass
            if not finished:
                self.root.after(100, self.poll)

        def view_history(self):
            win = tk.Toplevel(self.root)
            win.title("History")
            win.geometry("640x400")
            box = tk.Listbox(win, width=90, height=20)
            box.pack(pady=10)
            for item in self.history:
                box.insert(tk.END, item)

    root = tk.Tk()
    App(root)
    root.mainloop()


def build_parser():
    p = argparse.ArgumentParser(description="LRD-TECH Background Remover Pro")
    p.add_argument("-i", "--input", help="Single image file")
    p.add_argument("-d", "--directory", help="Folder for bulk processing")
    p.add_argument("-o", "--output", default="output", help="Output folder or file")
    p.add_argument("--engine", choices=["auto", "chroma", "hybrid", "ai"], default="auto")
    p.add_argument("--scale", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--spill", type=float, default=1.0)
    p.add_argument("--sharpen", type=float, default=0.7)
    p.add_argument("--denoise", type=float, default=3.0)
    p.add_argument("--feather", type=float, default=0.8)
    p.add_argument("--shrink", type=int, default=0)
    p.add_argument("--reach", type=float, default=0.10)
    p.add_argument("--trim", action="store_true")
    p.add_argument("--format", choices=["png", "webp", "tiff", "jpg"], default="png")
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--model", default="isnet-general-use")
    p.add_argument("--flatten", default="255,255,255")
    return p


def main():
    args = build_parser().parse_args()
    if not args.input and not args.directory:
        run_gui()
        return 0
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())