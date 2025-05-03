import cv2
import torch
from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO('D:/project_kuntal/Project/runs/detect/train/weights/best.pt')  # Load a pretrained model (recommended for inference)

# Load the image
image_path = r'D:\project_kuntal\Project\images\img-698.jpg'
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Visualize the results
for result in results:
    result.show()

# Save the image with detections
output_dir = r'D:\project_kuntal\Project\Result'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'img-698_result.jpg')

for result in results:
    result.plot(save=True, filename=output_path)
