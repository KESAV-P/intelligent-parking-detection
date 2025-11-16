import os
import shutil
import pandas as pd
from tqdm import tqdm

# Paths
BASE_DIR = "/Users/kesavp/shell intern/smart_parking_system"
CSV_PATH = f"{BASE_DIR}/archive/CNRParkEXT.csv"
PATCHES_DIR = f"{BASE_DIR}/archive/CNRPark-Patches-150x150"

OUTPUT_DIR = f"{BASE_DIR}/dataset/classification"
EMPTY_DIR = f"{OUTPUT_DIR}/EMPTY"
OCC_DIR = f"{OUTPUT_DIR}/OCCUPIED"

# Create output folders
os.makedirs(EMPTY_DIR, exist_ok=True)
os.makedirs(OCC_DIR, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

missing = 0
copied = 0

print("Processing images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):

    # CSV columns: path, filename, occupied
    image_path = os.path.join(PATCHES_DIR, row["path"], row["filename"])

    if not os.path.exists(image_path):
        missing += 1
        continue

    # Output destination
    if row["occupied"] == 1:
        dest = OCC_DIR
    else:
        dest = EMPTY_DIR

    shutil.copy(image_path, dest)
    copied += 1

print(f"\nCopied: {copied}")
print(f"Missing: {missing}")
print(f"Done! Classification dataset at: {OUTPUT_DIR}")