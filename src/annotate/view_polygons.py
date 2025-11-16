import cv2
import json
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "/Users/kesavp/shell_intern/smart_parking_system/slotimage.png"
JSON_PATH = "slots.json"
# -----------------------------

points = []
polygons = []


# Load previous polygons if exists
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, "r") as f:
        polygons = json.load(f)
    print(f"🔄 Loaded {len(polygons)} polygons from {JSON_PATH}")


# Mouse callback
def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"✔ Point added: {x, y}")


def draw_all(img):
    """Draw saved polygons + current points"""
    overlay = img.copy()

    # Draw saved polygons
    for poly in polygons:
        name = list(poly.keys())[0]
        pts = np.array(poly[name], np.int32)
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        # Label
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(
            overlay, name, (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    # Draw current poly (not saved yet)
    if len(points) > 1:
        cv2.polylines(overlay, [np.array(points)], False, (0, 0, 255), 2)

    return overlay


# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ ERROR: Could not load image. Fix IMAGE_PATH.")
    exit()

cv2.namedWindow("Polygon Annotator")
cv2.setMouseCallback("Polygon Annotator", click_event)

slot_id = len(polygons) + 1

print("\n🖱 HOW TO USE:")
print("• Click to add points")
print("• Press ENTER to close polygon")
print("• Press S to save polygon")
print("• Press C to clear current points")
print("• Press Q to quit\n")

while True:
    display = draw_all(img)
    cv2.imshow("Polygon Annotator", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 13:  # ENTER
        if len(points) >= 3:
            print("🔺 Polygon closed.")
            points.append(points[0])  # close polygon visually
        else:
            print("❌ Need at least 3 points")

    elif key == ord("s"):  # Save polygon
        if len(points) >= 3:
            polygons.append({f"slot_{slot_id}": points.copy()})
            slot_id += 1

            with open(JSON_PATH, "w") as f:
                json.dump(polygons, f, indent=4)

            print("💾 Saved polygon to slots.json")

            points.clear()
        else:
            print("❌ No polygon to save")

    elif key == ord("c"):  # Clear unsaved points
        points.clear()
        print("🧹 Cleared current points")

    elif key == ord("q"):  # Quit
        print("👋 Exiting")
        break

cv2.destroyAllWindows()