import onnxruntime as ort
import numpy as np
import cv2

# -----------------------------
# 1. RUN ONNX MODEL
# -----------------------------
def run_onnx(model_path, image_path, img_size=(640, 640)):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, img_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    input_tensor = img_rgb.transpose(2, 0, 1)
    input_tensor = input_tensor.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    outputs = session.run(None, {input_name: input_tensor})
    return outputs, img_resized


# -----------------------------
# 2. PROCESS RAW YOLO OUTPUT
# -----------------------------
def process_output(output, conf_threshold=0.25):
    predictions = output[0][0]  # shape: (N, 85)

    boxes = []
    for pred in predictions:
        conf = pred[4]
        if conf < conf_threshold:
            continue

        x, y, w, h = pred[0:4]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        cls = np.argmax(pred[5:])
        boxes.append([x1, y1, x2, y2, conf, cls])

    return boxes


# -----------------------------
# 3. IOU FUNCTION
# -----------------------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


# -----------------------------
# 4. NON-MAX SUPPRESSION (NMS)
# -----------------------------
def nms(boxes, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    final_boxes = []

    while boxes:
        best = boxes.pop(0)
        final_boxes.append(best)
        boxes = [b for b in boxes if iou(best, b) < iou_threshold]

    return final_boxes


# -----------------------------
# 5. DRAW RESULTS ON IMAGE
# -----------------------------
def draw_boxes(img, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box

        cv2.rectangle(img, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.putText(img, f"{cls}:{conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Detections", img)
    cv2.waitKey(0)


# -----------------------------
# 6. RUN EVERYTHING
# -----------------------------
model = r"C:\Yolo1\runs\detect\train\weights\best.onnx"
image = r"C:\Yolo1\image2.png"

raw_output, resized_img = run_onnx(model, image)
boxes = process_output(raw_output)
final_boxes = nms(boxes)

print("Final detections:", final_boxes)

draw_boxes(resized_img, final_boxes)
