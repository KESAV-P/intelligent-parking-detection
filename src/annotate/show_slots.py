import cv2
import json
import numpy as np
import os

# ---------- FIXED PATHS ----------
IMAGE_PATH = "/Users/kesavp/shell_intern/smart_parking_system/slotimage.png"
JSON_PATH  = "/Users/kesavp/shell_intern/smart_parking_system/slots/slots.json"
# ---------------------------------

print("Looking for:", JSON_PATH)
print("Exists:", os.path.exists(JSON_PATH))

# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ Could not load image:", IMAGE_PATH)
    exit()

# Load polygons
try:
    with open(JSON_PATH, "r") as f:
        polygons = json.load(f)
except FileNotFoundError:
    print("❌ ERROR: slots.json not found.")
    exit()

print("Loaded polygons:", len(polygons))

# Draw polygons
for i, poly in enumerate(polygons):
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    cx, cy = pts.mean(axis=0).astype(int)
    cv2.putText(img, f"Slot {i+1}", (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

cv2.imshow("Saved Slots", img)
cv2.waitKey(0)
cv2.destroyAllWindows()