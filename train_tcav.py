import os
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression

from src.model_foundation import ViTPneumothoraxClassifier

def load_concept_images(concept_dir):
    """Loads and returns all PNG images from a directory as PyTorch tensors."""
    if not os.path.exists(concept_dir):
        return []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image_tensors = []
    for f in sorted(os.listdir(concept_dir)):
        if f.endswith(".png"):
            img_path = os.path.join(concept_dir, f)
            img = Image.open(img_path).convert("RGB")
            image_tensors.append(transform(img))
            
    if image_tensors:
        return torch.stack(image_tensors)
    return []

def extract_cls_embeddings(model, image_tensors, device):
    """Passes images through ViT and extracts CLS token hidden state representations."""
    model.eval()
    embeddings = []
    
    # Process in batches to avoid GPU/CPU memory pressure
    batch_size = 4
    num_samples = len(image_tensors)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = image_tensors[i:i+batch_size].to(device)
            # HF base model can be accessed via base_model.model
            outputs = model.resnet_or_vit.base_model.model.vit(batch, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1] # shape: [batch, 197, 768]
            cls_token = last_hidden_state[:, 0, :] # shape: [batch, 768]
            embeddings.append(cls_token.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)

def main():
    print("Starting CAV (Concept Activation Vector) training...")
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load PyTorch ViT model checkpoint
    checkpoint_path = "models/best_seed_0.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model = ViTPneumothoraxClassifier.load_from_checkpoint(checkpoint_path)
    else:
        print("Trained model checkpoint not found. Using default ViT weights for CAV extraction.")
        model = ViTPneumothoraxClassifier()
        
    model.to(device)
    
    concepts_base_dir = os.path.join("data", "concepts")
    if not os.path.exists(concepts_base_dir):
        raise FileNotFoundError(
            f"Concept directory not found at {concepts_base_dir}. Please run generate_mock_data.py first."
        )
        
    # Load random negative concepts
    random_tensors = load_concept_images(os.path.join(concepts_base_dir, "random"))
    if len(random_tensors) == 0:
        raise ValueError("No random background concept images found.")
        
    print(f"Loaded {len(random_tensors)} random background concept images.")
    random_embeddings = extract_cls_embeddings(model, random_tensors, device)
    
    # Define active medical concepts to train
    medical_concepts = {
        "Pleural Line": "pleural_line",
        "Rib Shadow": "rib_shadow",
        "Mediastinum": "mediastinum"
    }
    
    tcav_classifiers = {}
    
    for concept_name, concept_folder in medical_concepts.items():
        print(f"\nTraining Concept Activation Vector for: {concept_name}...")
        concept_tensors = load_concept_images(os.path.join(concepts_base_dir, concept_folder))
        
        if len(concept_tensors) == 0:
            print(f"Warning: No concept images found for {concept_name}. Skipping.")
            continue
            
        concept_embeddings = extract_cls_embeddings(model, concept_tensors, device)
        
        # Prepare training data: Positive concept (1) vs. Random Background (0)
        X = np.concatenate([concept_embeddings, random_embeddings], axis=0)
        y = np.concatenate([np.ones(len(concept_embeddings)), np.zeros(len(random_embeddings))], axis=0)
        
        # Train Logistic Regression classifier to define CAV decision boundary
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        
        score = clf.score(X, y)
        print(f"Classifier validation accuracy: {score:.4f}")
        
        # Save CAV weights coefficients and intercepts
        tcav_classifiers[concept_name] = {
            "weight": clf.coef_[0].tolist(),  # Save as list for standard serialization compatibility
            "intercept": float(clf.intercept_[0])
        }
        
    # Save the CAV vectors to disk
    out_path = os.path.join("models", "tcav_classifiers.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(tcav_classifiers, f)
        
    print(f"\nCAV training completed. Classifiers serialized successfully to {out_path}")

if __name__ == "__main__":
    main()
