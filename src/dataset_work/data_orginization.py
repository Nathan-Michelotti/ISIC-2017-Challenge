import os
import shutil
from pathlib import Path
from random import seed, shuffle
import pandas as pd

# Set reproducibility seed
seed(42)

# Paths
source_dir = "/home/nmichelotti/Desktop/Nanu Project/data/ISIC-2017_Training_Data"
csv_path = "/home/nmichelotti/Desktop/Nanu Project/data/ISIC-2017_Training_Part3_GroundTruth.csv"
output_base = "/home/nmichelotti/Desktop/Nanu Project/data/formatted_data"

# Load CSV
df = pd.read_csv(csv_path)

# Create class-based lists from CSV
mel_ids = df[df["melanoma"] == 1.0]["image_id"].tolist()
sk_ids = df[df["seborrheic_keratosis"] == 1.0]["image_id"].tolist()

shuffle(mel_ids)
shuffle(sk_ids)

# 80/20 split
mel_train = mel_ids[:200]
mel_test = mel_ids[200:]
sk_train = sk_ids[:200]
sk_test = sk_ids[200:]

# Make sure output folders exist
for split in ["train", "test"]:
    for cls in ["melanoma", "seborrheic_keratosis"]:
        Path(os.path.join(output_base, split, cls)).mkdir(parents=True, exist_ok=True)

# Function to copy by ID
def copy_images(image_ids, dest_folder, class_name):
    for img_id in image_ids:
        src_path = os.path.join(source_dir, img_id + ".jpg")
        dest_path = os.path.join(output_base, dest_folder, class_name, img_id + ".jpg")
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"Copied {img_id} to {dest_folder}/{class_name}")
        else:
            print(f"Missing: {src_path}")

# Copy training images
copy_images(mel_train, "train", "melanoma")
copy_images(sk_train, "train", "seborrheic_keratosis")

# Copy testing images
copy_images(mel_test, "test", "melanoma")
copy_images(sk_test, "test", "seborrheic_keratosis")
