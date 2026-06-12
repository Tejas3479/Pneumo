import os
import argparse
import shutil
import torch
import torch.nn as nn
import numpy as np

from src.model import PneumothoraxClassifier
from src.model_foundation import ViTPneumothoraxClassifier
from src.model_medfound import MedicalFoundationClassifier

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

class MedFoundWrapper(nn.Module):
    """
    Wraps MedicalFoundationClassifier model to output both logits and the CLS token representation.
    """
    def __init__(self, classifier: MedicalFoundationClassifier):
        super().__init__()
        self.vision_model = classifier.vision_model
        self.classifier = classifier.classifier

    def forward(self, x):
        outputs = self.vision_model(x)
        
        # Extract features (CLS token or pooled output)
        if hasattr(outputs, "last_hidden_state"):
            features = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif isinstance(outputs, tuple):
            features = outputs[0][:, 0, :] if outputs[0].ndim == 3 else outputs[0]
        else:
            features = outputs
            
        logits = self.classifier(features)
        return logits, features

def export_onnx_model(model_type, ckpt_path, weights_path, output_onnx_path=None):
    """
    Programmatic helper to export a PyTorch checkpoint to ONNX format.
    """
    ckpt_dir = os.path.dirname(ckpt_path) or "models"
    ckpt_name = os.path.basename(ckpt_path)

    m_type = model_type.lower()
    if m_type == "vit":
        model_class = ViTPneumothoraxClassifier
        wrapper_class = ViTModelWrapper
        output_names = ["logits", "cls_token"]
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "cls_token": {0: "batch"}
        }
    elif m_type == "medfound":
        model_class = MedicalFoundationClassifier
        wrapper_class = MedFoundWrapper
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

    # Load PyTorch model
    model = model_class.load_from_checkpoint(ckpt_path)
    model.eval()

    # Wrap model
    wrapper = wrapper_class(model)
    wrapper.eval()

    # Define dummy input
    example_input = torch.randn(1, 3, 224, 224)

    # Output filename definition
    if output_onnx_path is None:
        if ckpt_name == "best.ckpt":
            onnx_name = "model.onnx"
        else:
            seed_id = ckpt_name.replace("best_seed_", "").replace(".ckpt", "")
            onnx_name = f"model_{seed_id}.onnx"
        onnx_path = os.path.join(ckpt_dir, onnx_name)
    else:
        onnx_path = output_onnx_path

    # Export via ONNX compiler
    print(f"Compiling ONNX model: {onnx_path}...")
    torch.onnx.export(
        wrapper,
        example_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print(f"ONNX model compiled successfully.")

    # Save classification linear head weights
    print(f"Saving FC weights to {weights_path}...")
    m_type = model_type.lower()
    if m_type == "vit":
        try:
            classifier = model.resnet_or_vit.base_model.model.classifier
        except AttributeError:
            classifier = model.resnet_or_vit.classifier
    elif m_type == "medfound":
        classifier = model.classifier
    else:
        classifier = model.resnet.fc

    fc_weights = {
        "weight": classifier.weight.detach().cpu().numpy(),
        "bias": classifier.bias.detach().cpu().numpy()
    }
    np.save(weights_path, fc_weights)
    print("FC layer weights saved successfully.")
    return onnx_path

def main():
    parser = argparse.ArgumentParser(description="Export Trained Model checkpoints to ONNX format")
    parser.add_argument("--model_type", type=str, default="vit", choices=["resnet", "vit", "medfound"], help="Type of model architecture")
    parser.add_argument("--weights_path", type=str, default="models/model_weights.npy", help="Path to save classification layer weights")
    args = parser.parse_args()

    # Find all trained checkpoint files in models/ directory matching seed indices
    ckpt_dir = "models"
    ckpt_files = []
    if os.path.exists(ckpt_dir):
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith("best_seed_") and f.endswith(".ckpt")]
    
    if len(ckpt_files) == 0:
        if os.path.exists(os.path.join(ckpt_dir, "best.ckpt")):
            ckpt_files = ["best.ckpt"]
        else:
            raise FileNotFoundError("No checkpoints found in models/ to export. Run train.py first.")

    print(f"Detected {len(ckpt_files)} checkpoints to export.")

    # Iterate and export each checkpoint
    for idx, ckpt_name in enumerate(sorted(ckpt_files)):
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"\n==========================================")
        print(f" EXPORTING CHECKPOINT {ckpt_name}")
        print(f"==========================================")
        
        export_onnx_model(args.model_type, ckpt_path, args.weights_path)

    # Copy primary exported model to models/model.onnx if only 1 model was compiled
    primary_onnx_path = os.path.join(ckpt_dir, "model_0.onnx")
    default_onnx_path = os.path.join(ckpt_dir, "model.onnx")
    if os.path.exists(primary_onnx_path) and not os.path.exists(default_onnx_path):
        shutil.copy(primary_onnx_path, default_onnx_path)
        print(f"Copied primary model_0.onnx to {default_onnx_path}")

    print("\nAll model checkpoints exported successfully.")

if __name__ == "__main__":
    main()
