import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from src.model import PneumothoraxClassifier

class ModelWrapper(nn.Module):
    """
    Wraps the classifier and registers a forward hook on resnet50.layer4[-1]
    to capture intermediate feature maps during inference and return them
    alongside the logits.
    """
    def __init__(self, classifier: PneumothoraxClassifier):
        super().__init__()
        self.resnet = classifier.resnet
        self.feature_map = None
        
        # Register forward hook on the last block of layer4
        # For ResNet-50, this is the last Bottleneck module in layer 4.
        self.hook = self.resnet.layer4[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # Capture the feature map tensor
        self.feature_map = output

    def forward(self, x):
        self.feature_map = None
        logits = self.resnet(x)
        return logits, self.feature_map

def main():
    parser = argparse.ArgumentParser(description="Export Trained Model to ONNX format")
    parser.add_argument("--checkpoint", type=str, default="models/best.ckpt", help="Path to best PyTorch Lightning checkpoint")
    parser.add_argument("--onnx_path", type=str, default="models/model.onnx", help="Path to write exported ONNX model")
    parser.add_argument("--weights_path", type=str, default="models/model_weights.npy", help="Path to save fully connected layer weights")
    args = parser.parse_args()

    # Fallback checkpoint detection if name contains epoch/step modifications
    if not os.path.exists(args.checkpoint):
        checkpoint_dir = os.path.dirname(args.checkpoint)
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")] if os.path.exists(checkpoint_dir) else []
        if ckpt_files:
            args.checkpoint = os.path.join(checkpoint_dir, ckpt_files[0])
            print(f"Checkpoint not found. Using auto-detected: {args.checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")

    # Load model from checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = PneumothoraxClassifier.load_from_checkpoint(args.checkpoint)
    model.eval()

    # Wrap model for double output
    wrapper = ModelWrapper(model)
    wrapper.eval()

    # Create dummy input: 1 batch, 3 channels, 224x224
    example_input = torch.randn(1, 3, 224, 224)

    # Export model to ONNX
    print(f"Exporting model to {args.onnx_path}...")
    torch.onnx.export(
        wrapper,
        example_input,
        args.onnx_path,
        export_params=True,
        opset_version=15,  # Using stable and modern ONNX opset
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits", "feature_map"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
            "feature_map": {0: "batch"}
        }
    )
    print("ONNX model exported successfully.")

    # Save classification layers weights and biases
    print(f"Saving FC weights to {args.weights_path}...")
    fc_weights = {
        "weight": model.resnet.fc.weight.detach().numpy(),
        "bias": model.resnet.fc.bias.detach().numpy()
    }
    np.save(args.weights_path, fc_weights)
    print("FC layer weights saved successfully.")

if __name__ == "__main__":
    main()
