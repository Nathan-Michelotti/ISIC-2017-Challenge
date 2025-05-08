import pandas as pd
import os
import shutil
from random import seed, sample

# Set a seed for reproducibility
seed(42)

# Path to your CSV
csv_path = "/home/nmichelotti/Desktop/Nanu Project/data/ISIC-2017_Training_Part3_GroundTruth.csv"
df = pd.read_csv(csv_path)

# Filter 250 melanoma and 250 seborrheic_keratosis cases
mel_ids = df[df["melanoma"] == 1.0]["image_id"].tolist()
sk_ids = df[df["seborrheic_keratosis"] == 1.0]["image_id"].tolist()

mel_sample = sample(mel_ids, min(250, len(mel_ids)))
sk_sample = sample(sk_ids, min(250, len(sk_ids)))

# Set of image_ids to keep
keep_ids = set(mel_sample + sk_sample)

# Folder with images
folder = "/home/nmichelotti/Desktop/Nanu Project/data/ISIC-2017_Training_Data"

# Loop through files and delete unwanted ones
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    
    # Always keep .csv files
    if filename.lower().endswith(".csv"):
        continue
    
    # Check if it's one of the 500 selected images
    image_id, ext = os.path.splitext(filename)
    if image_id not in keep_ids:
        os.remove(filepath)
        print(f"Deleted: {filepath}")
