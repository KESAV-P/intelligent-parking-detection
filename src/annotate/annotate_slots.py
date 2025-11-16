import cv2
import json
import numpy as np
import os

IMAGE_PATH = "slotimage.png"
SAVE_PATH = "slots/slots.json"   # ALWAYS save inside slots/

points = []
polygons = []

def mouse_event(event, x, y, flags, param):
    global points, polygons

    # LEFT CLICK → add point
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print("Point added:", (x, y))

    # RIGHT CLICK → close polygon
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 3:
            polygons.append(points.copy())
            print("Polygon saved:", points)
        else:
            print("❌ Need at least 3 points")
        points = []


def main():
    if not os.path.exists(IMAGE_PATH):
        print("❌ Image not found:", IMAGE_PATH)
        return

    img = cv2.imread(IMAGE_PATH)
    cv2.namedWindow("Annotate")
    cv2.setMouseCallback("Annotate", mouse_event)

    print("\n📌 HOW TO USE")
    print("• Left-click = add point")
    print("• Right-click = finish polygon")
    print("• S = save")
    print("• Q = quit\n")

    while True:
        canvas = img.copy()

        # draw polygons while editing
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points)], False, (0, 0, 255), 2)

        # draw saved polygons
        for poly in polygons:
            cv2.polylines(canvas, [np.array(poly)], True, (255, 0, 0), 2)

        cv2.imshow("Annotate", canvas)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            os.makedirs("slots", exist_ok=True)
            with open(SAVE_PATH, "w") as f:
                json.dump(polygons, f)
            print("💾 Saved to", SAVE_PATH)

        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()