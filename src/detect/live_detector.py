# src/detect/live_detector.py
import cv2
import json
import numpy as np
import os
import sys
import time
import tensorflow as tf

# ------------ CONFIG ------------
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
SLOTS_JSON = "slots/slots_scaled.json"  # scaled polygons
MODEL_PATH = "slot_classifier.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5
# --------------------------------


# Load slots.json
def load_slots(path):
    with open(path, "r") as f:
        polys = json.load(f)
    return [np.array(p, dtype=np.int32) for p in polys]


# Safe crop polygon region from frame
def crop_polygon(frame, poly):
    h, w = frame.shape[:2]

    poly = poly.copy()
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

    x, y, ww, hh = cv2.boundingRect(poly)
    if ww < 3 or hh < 3:
        return None

    roi = frame[y:y+hh, x:x+ww]
    if roi.size == 0:
        return None

    mask = np.zeros((hh, ww), dtype=np.uint8)
    shifted = poly - np.array([x, y])
    cv2.fillPoly(mask, [shifted], 255)

    if mask.shape[:2] != roi.shape[:2]:
        return None

    return cv2.bitwise_and(roi, roi, mask=mask)


# Preprocess for model
def preprocess_crop(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype("float32") / 255.0


# Draw overlay
def draw_overlay(frame, poly, label, score):
    color = (0, 255, 0) if label == "FREE" else (0, 0, 255)

    cv2.polylines(frame, [poly.reshape((-1, 1, 2))], True, color, 2)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    cx, cy = poly.mean(axis=0).astype(int)
    cv2.putText(frame, f"{label} {score:.2f}", (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():

    if not os.path.exists(VIDEO_PATH):
        print("❌ Video not found:", VIDEO_PATH)
        return

    if not os.path.exists(SLOTS_JSON):
        print("❌ slots.json not found:", SLOTS_JSON)
        return

    if not os.path.exists(MODEL_PATH):
        print("❌ slot_classifier.h5 not found:", MODEL_PATH)
        return

    print("📌 Loading polygons...")
    slots = load_slots(SLOTS_JSON)
    print("Loaded:", len(slots), "slots")

    print("📌 Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded!")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    print("🎥 Running detection... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🎬 Video ended.")
            break

        frame_draw = frame.copy()
        crops = []
        indexes = []

        # Preprocess all slots
        for i, poly in enumerate(slots):
            crop = crop_polygon(frame, poly)
            if crop is None:
                crops.append(None)
                continue
            crops.append(preprocess_crop(crop))
            indexes.append(i)

        preds = np.zeros(len(slots)) - 1

        if indexes:
            batch = np.array([crops[i] for i in indexes])
            pred_vals = model.predict(batch, verbose=0).reshape(-1)
            for idx, val in zip(indexes, pred_vals):
                preds[idx] = float(val)

        # Draw labels
        for i, poly in enumerate(slots):
            s = preds[i]
            if s < 0:
                draw_overlay(frame_draw, poly, "ERR", 0)
            else:
                label = "FREE" if s < THRESHOLD else "OCCUPIED"
                draw_overlay(frame_draw, poly, label, s)

        cv2.imshow("Live Parking Detector", frame_draw)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()