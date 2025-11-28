import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX(r"C:\Yolo1\runs\detect\train\weights\best.onnx")

# Dummy image
blob = np.random.rand(1, 3, 640, 640).astype(np.float32)
net.setInput(blob)

out = net.forward()
print("Output shape:", out.shape)


