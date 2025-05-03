# Road-Pothole-Detection-Project
The primary use case for this project is to assist in road maintenance by identifying and categorizing potholes from images. This can help in prioritizing repair work based on the severity of the potholes detected.






Pothole Detection Project Documentation

Project Overview
This project aims to detect and classify potholes in images using a trained YOLO (You Only Look Once) model. The detected potholes are categorized into three classes: major pothole, medium pothole, and minor pothole. The project processes images from a specified folder, saves annotated images, generates a CSV report with the counts of each pothole type, and creates a pie chart showing the distribution of potholes.
Use Case
The primary use case for this project is to assist in road maintenance by identifying and categorizing potholes from images. This can help in prioritizing repair work based on the severity of the potholes detected.
Step-by-Step Guide
1. Prerequisites

Python 3.6 or higher
Required Python libraries: opencv-python, torch, ultralytics, pandas, matplotlib, torchvision
2. Install Required Libraries
You can install the required libraries using pip:
pip install opencv-python torch ultralytics pandas matplotlib torchvision

3. Prepare the YOLO Model
Ensure you have a trained YOLO model. Place the model weights file (`best.pt`) in the appropriate directory.
4. Prepare the Image Folder
Place the images you want to process in a specified folder. Ensure the images are in .jpg, .jpeg, or .png format.
5. Create the Python Script
   
Create a Python script (e.g., pothole_detection.py) with the following code:

import os
import cv2
import torch
from ultralytics import YOLO
import pandas as pd
from torchvision.ops import nms
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO('D:/project_kuntal/Project/runs/detect/train/weights/best.pt')  # Load a pretrained model (recommended for inference)

# Define the image folder path
image_folder = r'D:\project_kuntal\Project\Project_code\data'
output_csv = r'D:\project_kuntal\Project\Project_code\pothole_detection_report.csv'

# Initialize a list to store the results
results_list = []

# Process each image in the folder
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        
        # Run inference
        results = model(image)
        
        # Initialize counters for each class
        major_pothole_count = 0
        medium_pothole_count = 0
        minor_pothole_count = 0
        
        # Collect bounding boxes, scores, and class names
        boxes = []
        scores = []
        class_names = []
        
        for result in results:
            for det in result.boxes:
                class_name = model.names[int(det.cls)]
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                score = det.conf.item()
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_names.append(class_name)
        
        # Convert lists to tensors
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        
        # Apply Non-Maximum Suppression (NMS)
        nms_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.7)
        
        # Filter boxes, scores, and class names based on NMS results
        nms_boxes = boxes_tensor[nms_indices].tolist()
        nms_class_names = [class_names[i] for i in nms_indices]
        
        # Draw bounding boxes and count the occurrences of each class
        for box, class_name in zip(nms_boxes, nms_class_names):
            x1, y1, x2, y2 = map(int, box)
            if class_name == 'major_pothole':
                major_pothole_count += 1
            elif class_name == 'medium_pothole':
                medium_pothole_count += 1
            elif class_name == 'minor_pothole':
                minor_pothole_count += 1
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Append the results to the list
        results_list.append([image_name, major_pothole_count, medium_pothole_count, minor_pothole_count])
        
        # Save the annotated image
        annotated_image_path = os.path.join(image_folder, f'annotated_{image_name}')
        cv2.imwrite(annotated_image_path, image)

# Create a DataFrame and save to CSV
df = pd.DataFrame(results_list, columns=['img_name', 'no_of_major_pothole', 'no_of_medium_pothole', 'no_of_minor_pothole'])
df.to_csv(output_csv, index=False)

print(f'Report saved to {output_csv}')

# Generate a pie chart
# Calculate the total counts for each category
total_major_pothole = df['no_of_major_pothole'].sum()
total_medium_pothole = df['no_of_medium_pothole'].sum()
total_minor_pothole = df['no_of_minor_pothole'].sum()

# Data for the pie chart
labels = ['Major Pothole', 'Medium Pothole', 'Minor Pothole']
sizes = [total_major_pothole, total_medium_pothole, total_minor_pothole]
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.1, 0, 0)  # explode the 1st slice (Major Pothole)

# Create the pie chart
plt.figure(figsize=(10, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Save the pie chart
output_pie_chart = r'D:\project_kuntal\Project\Project_code\pothole_distribution_pie_chart.png'
plt.savefig(output_pie_chart)

print(f'Pie chart saved to {output_pie_chart}')

6. Run the Script
Run the Python script to process the images, generate the CSV report, and create the pie chart:
python pothole_detection.py

7. Review the Results

The annotated images will be saved in the specified image folder with the prefix annotated_.
The CSV report will be saved at the specified path (`pothole_detection_report.csv`).
The pie chart will be saved at the specified path (`pothole_distribution_pie_chart.png`).
Conclusion
This project provides a comprehensive solution for detecting and categorizing potholes in images using a YOLO model. The results are saved in a CSV file and visualized in a pie chart, making it easier to analyze the distribution of potholes.
