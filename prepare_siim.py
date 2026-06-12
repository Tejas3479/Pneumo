import os
import pandas as pd
import pydicom
import numpy as np
import cv2
import argparse
import subprocess
import zipfile
from sklearn.model_selection import train_test_split

def download_siim():
    """Download dataset using Kaggle API and extract it using Python zipfile."""
    os.makedirs("data/raw", exist_ok=True)
    print("Downloading SIIM-ACR dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "siim-acr-pneumothorax-segmentation", "-p", "data/raw"], check=True)
    zip_path = os.path.join("data", "raw", "siim-acr-pneumothorax-segmentation.zip")
    extract_dir = os.path.join("data", "siim")
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete.")

def convert_and_split(frac_val=0.1, frac_test=0.1):
    """Convert DICOM → PNG and produce data/siim/train.csv, val.csv, test.csv"""
    csv_path = os.path.join("data", "siim", "train-rle.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"SIIM-ACR metadata not found at {csv_path}. Please run with --download first.")
        
    df = pd.read_csv(csv_path)  # columns: ImageId, EncodedPixels
    # Create label: 1 if any positive mask
    df["has_pneumo"] = df.groupby("ImageId")[" EncodedPixels"].transform(
        lambda x: (x != "-1").any().astype(int)
    )
    df_unique = df[["ImageId", "has_pneumo"]].drop_duplicates()

    png_dir = os.path.join("data", "siim", "png")
    os.makedirs(png_dir, exist_ok=True)
    
    print("Converting DICOM files to PNG...")
    # Walk through DICOM files and convert
    dicom_root = os.path.join("data", "siim", "dicom-images-train")
    for img_id in df_unique["ImageId"]:
        # The dicom files in SIIM ACR are nested under study/series folders, or in a flat layout
        # Let's search recursively for the DICOM file matching the ImageId
        dcm_file = None
        for root, dirs, files in os.walk(dicom_root):
            for file in files:
                if file.startswith(img_id) or file == f"{img_id}.dcm":
                    dcm_file = os.path.join(root, file)
                    break
            if dcm_file:
                break
                
        if dcm_file and os.path.exists(dcm_file):
            try:
                ds = pydicom.dcmread(dcm_file)
                img = ds.pixel_array.astype(np.float32)
                # Normalize to [0,255]
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
                cv2.imwrite(os.path.join(png_dir, f"{img_id}.png"), img.astype(np.uint8))
            except Exception as e:
                print(f"Failed to convert {img_id}: {e}")
        else:
            print(f"DICOM file for {img_id} not found under {dicom_root}")

    print("Splitting datasets...")
    # Split
    train, temp = train_test_split(df_unique, test_size=frac_val+frac_test, stratify=df_unique["has_pneumo"], random_state=42)
    val, test = train_test_split(temp, test_size=frac_test/(frac_val+frac_test), stratify=temp["has_pneumo"], random_state=42)

    for split_df, name in zip([train, val, test], ["train", "val", "test"]):
        out_csv = os.path.join("data", "siim", f"{name}.csv")
        split_df.to_csv(out_csv, index=False)
        print(f"Saved {name} split to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download raw dataset from Kaggle")
    args = parser.parse_args()
    if args.download:
        download_siim()
    convert_and_split()
