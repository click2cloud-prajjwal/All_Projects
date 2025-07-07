import os
import shutil
import random

def split_dataset(input_dir, output_dir, train_ratio=0.8):
    # Set seed for reproducibility
    random.seed(42)

    # Loop through each class folder
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # Get all image files in the class folder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(images)

        # Split into train and test
        split_index = int(len(images) * train_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Copy images to new train and test directories
        for split_type, split_images in [('train', train_images), ('test', test_images)]:
            split_class_dir = os.path.join(output_dir, split_type, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for image in split_images:
                src_path = os.path.join(class_path, image)
                dst_path = os.path.join(split_class_dir, image)
                shutil.copy2(src_path, dst_path)

    print("Dataset split completed: 80% train / 20% test")


split_dataset("/home/ubuntu/pests_and_disease/Rice_pest_detection/Dataset-rebuild", "/home/ubuntu/pests_and_disease/Rice_pest_detection/dataset")
