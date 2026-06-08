import os
import argparse
import shutil
import torch
import torch.nn as nn
import numpy as np

from src.model import PneumothoraxClassifier
from src.model_foundation import ViTPneumothoraxClassifier

class ResNetModelWrapper(nn.Module):
    """
    Wraps ResNet model to output both logits and feature maps.
    """
    def __init__(self, classifier: PneumothoraxClassifier):
        super().__init__()
        self.resnet = classifier.resnet
        self.feature_map = None
        self.hook = self.resnet.layer4[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.feature_map = output

    def forward(self, x):
        self.feature_map = None
        logits = self.resnet(x)
        return logits, self.feature_map

class ViTModelWrapper(nn.Module):
    """
    Wraps ViT model to output both logits and the CLS token representation.
    """
    def __init__(self, classifier: ViTPneumothoraxClassifier):
        super().__init__()
        # Retrieve the underlying HuggingFace base model
        self.vit = classifier.resnet_or_vit.base_model.model

    def forward(self, x):
        outputs = self.vit(x, output_hidden_states=True)
        logits = outputs.logits
        last_hidden_state = outputs.hidden_states[-1]  # shape: [batch, 197, 768]
        cls_token = last_hidden_state[:, 0, :]         # shape: [batch, 768]
        return logits, cls_token

def main():
    parser = argparse.ArgumentParser(description="Export Trained Model checkpoints to ONNX format")
    parser.add_argument("--model_type", type=str, default="vit", choices=["resnet", "vit"], help="Type of model architecture")
    parser.add_argument("--weights_path", type=str, default="models/model_weights.npy", help="Path to save classification layer weights")
    args = parser.parse_args()

    # Find all trained checkpoint files in models/ directory matching seed indices
    ckpt_dir = "models"
    ckpt_files = []
    if os.path.exists(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_seed_") and f.endswith(".ckpt")]
    
    if len(ckpt_files) == 0:
        # Fallback to general best.ckpt if seed check fails
        if os.path.exists(os.path.join(ckpt_dir, "best.ckpt")):
            ckpt_files = ["best.ckpt"]
        else:
            raise FileNotFoundError("No checkpoints found in models/ to export. Run train.py first.")

    print(f"Detected {len(ckpt_files)} checkpoints to export.")

    # Determine architecture class
    if args.model_type.lower() == "vit":
        model_class = ViTPneumothoraxClassifier
        wrapper_class = ViTModelWrapper
        output_names = ["logits", "cls_token"]
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "cls_token": {0: "batch"}
        }
    else:
        model_class = PneumothoraxClassifier
        wrapper_class = ResNetModelWrapper
        output_names = ["logits", "feature_map"]
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "feature_map": {0: "batch"}
        }

    # Iterate and export each checkpoint
    for idx, ckpt_name in enumerate(sorted(ckpt_files)):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"\n==========================================")
        print(f" EXPORTING CHECKPOINT {ckpt_name}")
        print(f"==========================================")

        # Load PyTorch model
        model = model_class.load_from_checkpoint(ckpt_path)
        model.eval()

        # Wrap model
        wrapper = wrapper_class(model)
        wrapper.eval()

        # Define dummy input
        example_input = torch.randn(1, 3, 224, 224)

        # Output filename definition
        if ckpt_name == "best.ckpt":
            onnx_name = "model.onnx"
        else:
            # Map best_seed_X.ckpt -> model_X.onnx
            seed_id = ckpt_name.replace("best_seed_", "").replace(".ckpt", "")
            onnx_name = f"model_{seed_id}.onnx"

        onnx_path = os.path.join(ckpt_dir, onnx_name)

        # Export via ONNX compiler
        print(f"Compiling ONNX model: {onnx_path}...")
        torch.onnx.export(
            wrapper,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=18,  # Using stable and modern ONNX opset for Python 3.13
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print(f"Checkpoint {ckpt_name} compiled successfully.")

        # Save classification linear head weights for the primary model (seed 0)
        if idx == 0 or ckpt_name == "best.ckpt":
            print(f"Saving primary FC weights to {args.weights_path}...")
            if args.model_type.lower() == "vit":
                # PEFT wraps classifier head in modules_to_save
                # The parameters are stored in base_model.model.classifier
                try:
                    classifier = model.resnet_or_vit.base_model.model.classifier
                except AttributeError:
                    classifier = model.resnet_or_vit.classifier
            else:
                classifier = model.resnet.fc

            fc_weights = {
                "weight": classifier.weight.detach().numpy(),
                "bias": classifier.bias.detach().numpy()
            }
            np.save(args.weights_path, fc_weights)
            print("FC layer weights saved successfully.")

    # Copy primary exported model to models/model.onnx if only 1 model was compiled
    primary_onnx_path = os.path.join(ckpt_dir, "model_0.onnx")
    default_onnx_path = os.path.join(ckpt_dir, "model.onnx")
    if os.path.exists(primary_onnx_path) and not os.path.exists(default_onnx_path):
        shutil.copy(primary_onnx_path, default_onnx_path)
        print(f"Copied primary model_0.onnx to {default_onnx_path}")

    print("\nAll model checkpoints exported successfully.")

if __name__ == "__main__":
    main()
