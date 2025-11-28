from ultralytics import YOLO
#Run inference on an image
#yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt,yolo11x.pt
model = YOLO("C:/Yolo1/env2/yolov8x.pt")
results=model("vid2.mp4",stream=True)
#process results list
for result in results:
    boxes=result.boxes
    masks=result.masks
    keypoints=result.keypoints
    probs=result.probs
    obb=result.obb
    result.save(filename="result.jpg")