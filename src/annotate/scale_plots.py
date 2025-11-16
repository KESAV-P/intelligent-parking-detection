import json
import numpy as np

# INPUT / OUTPUT
IN_JSON  = "slots/slots.json"
OUT_JSON = "slots/slots_scaled.json"

# IMAGE SIZE YOU ANNOTATED ON
IMG_W = 3388
IMG_H = 1900

# VIDEO SIZE
VID_W = 1920
VID_H = 1080

scale_x = VID_W / IMG_W
scale_y = VID_H / IMG_H

print("Scale X:", scale_x)
print("Scale Y:", scale_y)

with open(IN_JSON, "r") as f:
    polys = json.load(f)

new_polys = []

for poly in polys:
    arr = np.array(poly, dtype=float)
    arr[:,0] *= scale_x  # scale X
    arr[:,1] *= scale_y  # scale Y
    new_polys.append(arr.astype(int).tolist())

with open(OUT_JSON, "w") as f:
    json.dump(new_polys, f, indent=4)

print("✅ Scaled polygons saved to:", OUT_JSON)