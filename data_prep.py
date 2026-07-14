# data_prep.py
"""
Dataset helper for doodle_ai

- USE_SYNTHETIC: when True, generate synthetic stroke-like 28x28 bitmaps for each category.
- When False, follow the printed curl commands (or run download_real_data()) to fetch the
  real Quick, Draw! numpy bitmap files into data/<category>.npy.
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import random
import textwrap

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CATEGORIES = ["cat", "house", "tree", "car", "fish", "star", "umbrella", "banana", "bicycle", "clock"]

# Toggle: set False on your machine when you want to download actual Quick, Draw! .npy files
USE_SYNTHETIC = True

def print_download_instructions():
    print("Download each category's numpy bitmap (.npy) from Google's Quick, Draw! dataset.")
    print("Example curl commands (run on your machine where storage.googleapis.com is reachable):\n")
    for category in CATEGORIES:
        fname = f"{category}.npy"
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
        print(f"mkdir -p {DATA_DIR}")
        print(f"curl -L -o {DATA_DIR / fname} {url}")
    print("\nOr in Python:\n")
    print(textwrap.dedent(f"""\
        import requests
        from pathlib import Path
        DATA_DIR = Path("{DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for category in {CATEGORIES}:
            url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{{category}}.npy"
            r = requests.get(url)
            with open(DATA_DIR / f"{{category}}.npy", "wb") as f:
                f.write(r.content)
    """))

def generate_synthetic_for_category(category, n_samples=2000, img_size=28):
    arrs = []
    for i in range(n_samples):
        im = Image.new("L", (img_size, img_size), color=0)  # black
        draw = ImageDraw.Draw(im)
        # draw a small number of random strokes to make a doodle-like shape
        n_strokes = random.randint(1, 5)
        for s in range(n_strokes):
            points = []
            n_pts = random.randint(2, 6)
            for p in range(n_pts):
                x = random.randint(0, img_size - 1)
                y = random.randint(0, img_size - 1)
                points.append((x, y))
            width = random.randint(1, 3)
            draw.line(points, fill=255, width=width)
            # occasional circle/ellipse
            if random.random() < 0.2:
                x0 = random.randint(0, img_size - 6)
                y0 = random.randint(0, img_size - 6)
                x1 = x0 + random.randint(3, 8)
                y1 = y0 + random.randint(3, 8)
                draw.ellipse([x0, y0, x1, y1], outline=255)
        arr = np.asarray(im, dtype=np.uint8)
        arrs.append(arr)
    return np.stack(arrs, axis=0)

def save_synthetic(n_per_class=2000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        print(f"Generating {n_per_class} synthetic samples for {cat} ...")
        arr = generate_synthetic_for_category(cat, n_samples=n_per_class)
        outfile = DATA_DIR / f"{cat}.npy"
        np.save(outfile, arr)
        print(f"Saved {outfile} (shape {arr.shape})")

def validate_real_files():
    missing = []
    for cat in CATEGORIES:
        if not (DATA_DIR / f"{cat}.npy").exists():
            missing.append(cat)
    if missing:
        print("Missing .npy files for:", missing)
        print("Use print_download_instructions() to get curl/python commands.")
    else:
        print("All .npy files present in", DATA_DIR)

if __name__ == "__main__":
    if USE_SYNTHETIC:
        save_synthetic(n_per_class=2000)
        print("Synthetic data created in", DATA_DIR)
    else:
        print_download_instructions()
        validate_real_files()
