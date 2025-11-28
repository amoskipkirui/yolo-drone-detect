import cv2
import numpy as np

model_path = r"C:\Yolo1\runs\detect\train\weights\best.onnx"
image_path = r"C:\Yolo1\image2.png"

# Load ONNX model
net = cv2.dnn.readNetFromONNX(model_path)

# Load original image
img = cv2.imread(image_path)
orig_h, orig_w = img.shape[:2]

# ---- 1. Preprocess with letterbox (exact YOLO method) ---- #
input_size = 640
scale = min(input_size / orig_w, input_size / orig_h)

new_w = int(orig_w * scale)
new_h = int(orig_h * scale)

pad_w = input_size - new_w
pad_h = input_size - new_h

# resize
resized = cv2.resize(img, (new_w, new_h))

# pad equally left-right / top-bottom
pad_left = pad_w // 2
pad_top = pad_h // 2

img_padded = cv2.copyMakeBorder(
    resized,
    pad_top,
    pad_h - pad_top,
    pad_left,
    pad_w - pad_left,
    cv2.BORDER_CONSTANT,
    value=(114, 114, 114)
)

blob = cv2.dnn.blobFromImage(img_padded, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)

# ---- 2. Run inference ---- #
outs = net.forward()[0]

conf_threshold = 0.4
nms_threshold = 0.5

boxes = []
confidences = []
class_ids = []

# ---- 3. Decode YOLOv8 predictions ---- #
for det in outs:
    obj_conf = det[4]
    if obj_conf < conf_threshold:
        continue

    class_scores = det[5:]
    class_id = np.argmax(class_scores)
    cls_conf = class_scores[class_id]

    if cls_conf < conf_threshold:
        continue

    cx, cy, w_box, h_box = det[:4]

    # Convert xywh → x1y1x2y2
    x1 = cx - w_box / 2
    y1 = cy - h_box / 2
    x2 = cx + w_box / 2
    y2 = cy + h_box / 2

    # ---- 4. Undo letterbox padding & scaling ---- #
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale

    # store final box
    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    confidences.append(float(cls_conf))
    class_ids.append(class_id)

# ---- 5. NMS ---- #
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

if len(indices) == 0:
    print("No objects detected!")
else:
    for idx in indices:
        i = int(idx)
        x, y, w_box, h_box = boxes[i]
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(img, f"Class {class_ids[i]} {confidences[i]:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
