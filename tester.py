import os
from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

# Optional: suppress symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# === Load dataset ===
dataset = load_dataset("pathikg/drone-detection-dataset")

# === Output root folder ===
ROOT_DIR = "drone_yolo"

# Create YOLO-style folder structure
for split in ['train', 'val']:
    os.makedirs(os.path.join(ROOT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'labels', split), exist_ok=True)

# === Helper to convert bbox to YOLO format ===
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w /= img_width
    h /= img_height
    return [0, x_center, y_center, w, h]  # 0 = class ID for 'drone'

# === Convert each sample ===
for split in ['train', 'test']:
    yolo_split = 'val' if split == 'test' else split
    data = dataset[split]

    for idx, item in tqdm(enumerate(data), total=len(data), desc=f"Processing {split}"):
        image: Image.Image = item['image']
        img_width, img_height = image.size
        bboxes = item['objects']['bbox']

        # Save image
        image_filename = f"{split}_{idx:05d}.jpg"
        image_path = os.path.join(ROOT_DIR, 'images', yolo_split, image_filename)
        image.save(image_path)

        # Save label
        label_filename = f"{split}_{idx:05d}.txt"
        label_path = os.path.join(ROOT_DIR, 'labels', yolo_split, label_filename)

        with open(label_path, 'w') as f:
            for bbox in bboxes:
                yolo_box = convert_bbox_to_yolo(bbox, img_width, img_height)
                f.write(" ".join(f"{v:.6f}" for v in yolo_box) + "\n")
