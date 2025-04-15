import os
from collections import Counter
import matplotlib.pyplot as plt

data_dir = '/home/ubuntu/sugarcane_disease_detection/balanced_dataset'  # change to your dataset path
class_counts = {}

for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)
    if os.path.isdir(cls_path):
        count = len([img for img in os.listdir(cls_path) if img.lower().endswith(('.jpg', '.png', '.jpeg'))])
        class_counts[cls] = count

# Print class distribution
for k, v in class_counts.items():
    print(f"{k}: {v} images")

# Plot
plt.figure(figsize=(12, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()


# print("--------------------------------------------------------------------------------------------------------")


# import os
# import shutil
# import random
# from PIL import Image
# from torchvision import transforms
# from tqdm import tqdm

# # Paths
# src_root = 'Sugarcane_data'
# balanced_root = 'balanced_dataset'
# os.makedirs(balanced_root, exist_ok=True)

# # Your class counts
# class_counts = {
#     'Pokkah Boeng': 297,
#     'Rust': 514,
#     'RedRot': 618,
#     'Yellow': 505,
#     'smut': 316,
#     'Healthy': 622,
#     'Dried Leaves': 343,
#     'Grassy shoot': 346,
#     'Mosaic': 462,
#     'BrownRust': 314
# }

# max_count = max(class_counts.values())  # 622

# # Augmentation: horizontal flip + rotation + blur
# augment = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
# ])

# def augment_and_save(img_path, save_path, aug_count):
#     image = Image.open(img_path).convert('RGB')
#     for i in range(aug_count):
#         augmented = augment(image)
#         base_name = os.path.splitext(os.path.basename(img_path))[0]
#         new_name = f"{base_name}_aug{i}.jpg"
#         augmented.save(os.path.join(save_path, new_name))

# # Balance each class
# for cls, count in tqdm(class_counts.items(), desc="Balancing classes"):
#     src_class_dir = os.path.join(src_root, cls)
#     dst_class_dir = os.path.join(balanced_root, cls)
#     os.makedirs(dst_class_dir, exist_ok=True)

#     # Copy original images
#     image_files = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     for f in image_files:
#         shutil.copy(os.path.join(src_class_dir, f), dst_class_dir)

#     # Oversample with augmentation
#     need = max_count - count
#     if need > 0:
#         sampled = random.choices(image_files, k=need)
#         for i, f in enumerate(sampled):
#             src_path = os.path.join(src_class_dir, f)
#             augment_and_save(src_path, dst_class_dir, 1)

# print("✅ Done! All classes balanced to", max_count, "images each.")
# print("📁 Output saved to:", balanced_root)
