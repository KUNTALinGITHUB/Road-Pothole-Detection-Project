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
