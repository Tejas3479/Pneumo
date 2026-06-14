"""
Downloads and caches all pretrained model weights locally for offline / air-gapped operation.
Run ONCE before training when internet is available:

    python download_pretrained.py

Saves to:
    models/pretrained/vit-base-patch16-224-in21k/   ← ViT weights + config + feature_extractor
    models/pretrained/resnet50.pth                  ← ResNet-50 ImageNet weights
"""
import os
import torch
import torchvision.models as models
from transformers import ViTForImageClassification, ViTConfig, ViTFeatureExtractor


def download_all():
    print("Creating models/pretrained/ directory...")
    os.makedirs(os.path.join("models", "pretrained"), exist_ok=True)

    # 1. Download ViT model, config and feature extractor, save locally
    vit_dir = os.path.join("models", "pretrained", "vit-base-patch16-224-in21k")
    print(f"\nDownloading google/vit-base-patch16-224-in21k to {vit_dir}...")
    try:
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.num_labels = 1
        config.output_hidden_states = True
        config.output_attentions = True
        config.save_pretrained(vit_dir)

        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            config=config,
            ignore_mismatched_sizes=True
        )
        model.save_pretrained(vit_dir)

        # Also save the feature extractor (image processor) for inference preprocessing
        try:
            feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
            feature_extractor.save_pretrained(vit_dir)
            print(f"ViT feature extractor saved to {vit_dir}")
        except Exception as fe_err:
            print(f"ViT feature extractor download failed (non-fatal): {fe_err}")

        print(f"ViT model and config saved to {vit_dir}")
    except Exception as e:
        print(f"Error downloading/saving ViT: {e}")

    # 2. Download ResNet-50 weights and save state_dict locally
    resnet_path = os.path.join("models", "pretrained", "resnet50.pth")
    print(f"\nDownloading ResNet-50 weights to {resnet_path}...")
    try:
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)
        torch.save(resnet.state_dict(), resnet_path)
        print(f"ResNet-50 weights saved to {resnet_path}")
    except Exception as e:
        print(f"Error downloading/saving ResNet-50: {e}")

    print("\nAll pretrained weights downloaded successfully.")
    print("You can now train offline using:")
    print("  python train.py --model_type vit --epochs 20")


if __name__ == "__main__":
    download_all()
