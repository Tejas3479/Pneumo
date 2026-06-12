import os
import torch
import torchvision.models as models
from transformers import ViTForImageClassification, ViTConfig

def download_all():
    print("Creating models/pretrained/ directory...")
    os.makedirs(os.path.join("models", "pretrained"), exist_ok=True)
    
    # 1. Download ViT model & config, and save pretrained locally
    vit_dir = os.path.join("models", "pretrained", "vit-base-patch16-224-in21k")
    print(f"Downloading google/vit-base-patch16-224-in21k to {vit_dir}...")
    try:
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.save_pretrained(vit_dir)
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)
        model.save_pretrained(vit_dir)
        print("Successfully saved ViT model and config locally.")
    except Exception as e:
        print(f"Error downloading/saving ViT: {e}")
        
    # 2. Download ResNet-50 weights and save state_dict locally
    resnet_path = os.path.join("models", "pretrained", "resnet50.pth")
    print(f"Downloading ResNet-50 weights to {resnet_path}...")
    try:
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)
        torch.save(resnet.state_dict(), resnet_path)
        print("Successfully saved ResNet-50 weights locally.")
    except Exception as e:
        print(f"Error downloading/saving ResNet-50: {e}")

if __name__ == "__main__":
    download_all()
