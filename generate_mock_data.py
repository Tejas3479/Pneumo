import os
import random
import numpy as np
import pandas as pd
import pydicom
import cv2
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

def create_mock_dicom(filename, image_data):
    """
    Creates a mock DICOM file with standard headers and the provided 2D pixel array.
    """
    # Create the FileMetaDataset
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.1')  # Computed Radiography Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create the Dataset
    ds = Dataset()
    ds.file_meta = file_meta
    ds.preamble = b"\0" * 128

    # Add standard metadata
    ds.PatientName = "Test^Patient"
    ds.PatientID = f"PT-{random.randint(1000, 9999)}"
    ds.Modality = "DX"
    ds.StudyDate = "20260608"
    ds.StudyTime = "120000"
    
    # Image properties
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = image_data.shape[0]
    ds.Columns = image_data.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = "1.0"
    ds.RescaleIntercept = "0.0"

    # Scale pixel data to uint16
    pixel_array = (image_data * 60000).astype(np.uint16)
    ds.PixelData = pixel_array.tobytes()

    # Set file layout parameters
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Save to file
    ds.save_as(filename, write_like_original=False)

def generate_concepts(data_dir):
    """
    Generates 20 mock images per clinical concept class under data/concepts/
    using OpenCV drawing functions for TCAV training.
    """
    concepts = ["pleural_line", "rib_shadow", "mediastinum", "random"]
    print("Generating synthetic concept datasets for TCAV...")
    
    for concept in concepts:
        concept_dir = os.path.join(data_dir, "concepts", concept)
        os.makedirs(concept_dir, exist_ok=True)
        
        for idx in range(20):
            # Create black 3-channel image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            if concept == "pleural_line":
                # Draw a sharp white diagonal line to simulate a pleural edge
                x1 = random.randint(20, 80)
                y1 = random.randint(20, 80)
                x2 = random.randint(140, 200)
                y2 = random.randint(140, 200)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)
                
            elif concept == "rib_shadow":
                # Draw horizontal stripes simulating rib shadows
                for y_coord in range(20, 224, 45):
                    offset = random.randint(-6, 6)
                    cv2.rectangle(
                        img, 
                        (0, y_coord + offset), 
                        (224, y_coord + 20 + offset), 
                        (120, 120, 120), 
                        thickness=-1
                    )
                    
            elif concept == "mediastinum":
                # Draw a central light-grey ellipse representing the mediastinal shadow
                cx = 112 + random.randint(-15, 15)
                cy = 112 + random.randint(-15, 15)
                rx = random.randint(35, 55)
                ry = random.randint(70, 100)
                cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (200, 200, 200), thickness=-1)
                
            elif concept == "random":
                # Random Gaussian noise
                noise = np.random.rand(224, 224, 3) * 255
                img = np.uint8(noise)

            # Save as PNG
            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(concept_dir, f"concept_{idx:03d}.png"))
            
    print("Concept datasets created under data/concepts/")

def main():
    # Setup directories
    data_dir = "data"
    dicoms_dir = os.path.join(data_dir, "dicoms")
    os.makedirs(dicoms_dir, exist_ok=True)
    
    num_samples = 100
    records = []

    print(f"Generating {num_samples} mock DICOM images...")
    
    for i in range(num_samples):
        # Create a synthetic image structure: background noise + optional simulated lung/mass shape
        image_data = np.random.rand(224, 224) * 0.1  # background noise
        
        # Add basic geometric shapes to represent chest structure
        x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
        # Simulated left lung field
        lung_left = (x + 0.35)**2 + (y * 0.8)**2 < 0.25
        # Simulated right lung field
        lung_right = (x - 0.35)**2 + (y * 0.8)**2 < 0.25
        
        image_data[lung_left] += 0.5
        image_data[lung_right] += 0.5
        
        # Add some random structure/edges to make it look slightly like a lung X-ray
        image_data += np.exp(-((x)**2 + (y)**2) / 0.1) * 0.2  # mediastinum area
        
        # Clip to [0, 1] range
        image_data = np.clip(image_data, 0.0, 1.0)
        
        filename = f"image_{i:03d}.dcm"
        filepath = os.path.join(dicoms_dir, filename)
        create_mock_dicom(filepath, image_data)
        
        # 80% negative, 20% positive class bias
        label = 1 if random.random() < 0.20 else 0
        records.append({
            "ImagePath": f"dicoms/{filename}",
            "Label": label
        })
        
    # Create CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(data_dir, "train.csv")
    df.to_csv(csv_path, index=False)
    print(f"Mock data generation complete. CSV file written to {csv_path}")

    # Generate synthetic concept dataset
    generate_concepts(data_dir)

if __name__ == "__main__":
    main()
