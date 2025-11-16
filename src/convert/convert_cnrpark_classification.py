import os
import shutil
import pandas as pd
from tqdm import tqdm

# Path to dataset root
DATASET_ROOT = "/Users/kesavp/shell intern/archive"

CSV_PATH = os.path.join(DATASET_ROOT, "CNRParkEXT.csv")

OUTPUT_DIR = "dataset/classification"
FREE_DIR = os.path.join(OUTPUT_DIR, "free")
OCC_DIR = os.path.join(OUTPUT_DIR, "occupied")

# Known folders that actually exist
SEARCH_FOLDERS = [
    "CNR-EXT_FULL_IMAGE_1000x750",
    "CNR-EXT-Patches-150x150",
    "CNRPark-Patches-150x150"
]

def find_image(filename):
    """Search dataset folders to locate an image."""
    for folder in SEARCH_FOLDERS:
        folder_path = os.path.join(DATASET_ROOT, folder)
        for root, _, files in os.walk(folder_path):
            if filename in files:
                return os.path.join(root, filename)
    return None

def convert():
    os.makedirs(FREE_DIR, exist_ok=True)
    os.makedirs(OCC_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    copied = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
        img_path = row["image_url"]
        occupancy = row["occupancy"]  # 1 busy, 0 free

        filename = os.path.basename(img_path)

        real_path = find_image(filename)

        if real_path is None:
            missing += 1
            continue

        dest = os.path.join(FREE_DIR if occupancy == 0 else OCC_DIR, filename)

        shutil.copy(real_path, dest)
        copied += 1

    print(f"\nCopied: {copied} images")
    print(f"Missing: {missing} images")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert()