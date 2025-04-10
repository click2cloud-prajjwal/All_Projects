import cv2
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# --- Load image ---
image_path = "D:\Weed_and_Disease_Detector\Rice_Weed_Detection\stock-photo-marselia-quadrifolia-rice-weed-in-field-2508001651.jpg"
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- Load YOLO model (replace with your trained model path) ---
yolo_model = YOLO("D:/Weed_and_Disease_Detector/Rice_Weed_Detection/best.pt")  # or your custom model: "runs/detect/train/weights/best.pt"
results = yolo_model(image_rgb)[0]

# --- Weed class index ---
weed_class_id = 0  # change this to match your class

# --- Extract bounding boxes for weeds only ---
boxes = []
for box in results.boxes:
    if int(box.cls[0]) == weed_class_id:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        boxes.append([x1, y1, x2, y2])

# --- Load SAM model ---
sam_checkpoint = "D:\Weed_and_Disease_Detector\Rice_Weed_Detection\sam_vit_h_4b8939.pth"  # update this to your actual checkpoint
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)

# --- Generate combined mask for all YOLO boxes using SAM ---
composite_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

for box in boxes:
    masks, _, _ = predictor.predict(box=np.array(box)[None, :], multimask_output=False)
    mask = masks[0].astype(np.uint8)
    composite_mask = np.maximum(composite_mask, mask)

# --- Apply transparent reddish overlay where mask = 1 ---
highlight_color = (50, 50, 255)  # Soft red (BGR)
alpha = 0.35  # Transparency factor

# Create color overlay
color_mask = np.zeros_like(image_bgr)
for i in range(3):
    color_mask[:, :, i] = highlight_color[i]

# Blend image and mask
masked_overlay = cv2.addWeighted(image_bgr, 1.0, color_mask, alpha, 0)
final_image = np.where(composite_mask[:, :, None] == 1, masked_overlay, image_bgr)

# --- Optional: draw black contour around the weed area ---
contours, _ = cv2.findContours(composite_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(final_image, contours, -1, (0, 0, 0), 1)  # black thin outline

# --- Save or display ---
output_path = "weed_highlighted_red.png"
cv2.imwrite(output_path, final_image)
print(f"Saved highlighted image to: {output_path}")
