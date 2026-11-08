from pathlib import Path
from collections import defaultdict
import re
import shutil
from difflib import SequenceMatcher
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

SIMILARITY_THRESHOLD = 0.75
FRAME_NUM_PATTERN = re.compile(r"[_-]?\d+$")


def base_key(filename: str) -> str:
    """Return a normalized 'base name' for grouping PNGs."""
    stem = Path(filename).stem.lower()
    original_stem = stem
    stem = FRAME_NUM_PATTERN.sub("", stem)  # remove trailing frame numbers
    stem = re.sub(r"[\s_-]+", "_", stem)    # normalize separators
    cleaned = stem.strip("_")
    return cleaned if cleaned else original_stem


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class ImageConsolidatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PNG Folder Consolidator")
        self.root.geometry("650x500")
        self.root.minsize(500, 400)

        self.dir_path = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.LabelFrame(self.root, text="Select Target Directory", padx=10, pady=10)
        top_frame.pack(fill="x", padx=15, pady=15)

        self.entry_path = tk.Entry(top_frame, textvariable=self.dir_path, width=55, font=("Arial", 10))
        self.entry_path.pack(side="left", padx=(0, 10), expand=True, fill="x")

        browse_btn = tk.Button(top_frame, text="Browse...", command=self.browse_directory, bg="#e0e0e0", font=("Arial", 10, "bold"))
        browse_btn.pack(side="right")

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=(0, 10))

        self.run_btn = tk.Button(btn_frame, text="Run Consolidation", command=self.start_consolidation_thread, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), padx=10, pady=5)
        self.run_btn.pack(fill="x")

        log_frame = tk.LabelFrame(self.root, text="Consolidation Logs", padx=10, pady=10)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.log_area.pack(fill="both", expand=True)
        self.log_area.config(state="disabled")

    def log(self, message: str):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def browse_directory(self):
        selected_dir = filedialog.askdirectory()
        if selected_dir:
            self.dir_path.set(selected_dir)

    def start_consolidation_thread(self):
        path_str = self.dir_path.get().strip()
        if not path_str:
            messagebox.showerror("Error", "Please select a directory first.")
            return

        target_path = Path(path_str)
        if not target_path.exists() or not target_path.is_dir():
            messagebox.showerror("Error", "The specified path is not a valid directory.")
            return

        self.run_btn.config(state="disabled", bg="#9e9e9e")
        self.log_area.config(state="normal")
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state="disabled")

        threading.Thread(target=self.run_consolidation, args=(target_path,), daemon=True).start()

    def run_consolidation(self, dir_path: Path):
        try:
            self.log(f"Scanning directory: {dir_path.absolute()}")
            
            # FIXED: Collect all PNG files directly from the selected folder
            png_files = [p for p in dir_path.glob("*.png") if p.is_file()]

            if not png_files:
                self.log("No PNG files found in the selected directory.")
                messagebox.showinfo("Finished", "No PNG files found.")
                return

            self.log(f"Found {len(png_files)} PNG files to group.")

            # 2. Group PNGs by similar base names
            groups = []
            for png in png_files:
                key = base_key(png.name)
                placed = False
                for group in groups:
                    group_key = base_key(group[0].name)
                    if group_key == key or similar(key, group_key) >= SIMILARITY_THRESHOLD:
                        group.append(png)
                        placed = True
                        break
                if not placed:
                    groups.append([png])

            # 3. Move each group into its own folder named after the common base
            for group in groups:
                key_counts = defaultdict(int)
                for p in group:
                    key_counts[base_key(p.name)] += 1
                canonical = max(key_counts, key=key_counts.get) or "misc"

                target_dir = dir_path / canonical
                target_dir.mkdir(exist_ok=True)

                for png in group:
                    dest = target_dir / png.name
                    if dest.exists() and dest.resolve() != png.resolve():
                        counter = 1
                        while True:
                            new_name = f"{png.stem}_{counter}{png.suffix}"
                            dest = target_dir / new_name
                            if not dest.exists():
                                break
                            counter += 1

                    shutil.move(str(png), str(dest))
                    self.log(f"Moved: {png.name} -> {canonical}/")

            self.log("\nConsolidation complete successfully!")
            messagebox.showinfo("Success", "Consolidation complete!")

        except Exception as e:
            self.log(f"An error occurred: {str(e)}")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

        finally:
            self.run_btn.config(state="normal", bg="#4CAF50")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageConsolidatorApp(root)
    root.mainloop()