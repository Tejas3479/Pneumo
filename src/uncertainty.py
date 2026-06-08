import os
import numpy as np
import onnxruntime as ort

class EnsemblePredictor:
    """
    Loads up to 5 ONNX models trained under different seeds, runs ensemble inference,
    and computes prediction mean probabilities alongside standard deviation uncertainty metrics.
    """
    def __init__(self, models_dir: str = "models"):
        self.sessions = []
        
        # Check for seed-specific ensemble models: model_0.onnx, model_1.onnx...
        for i in range(5):
            model_path = os.path.join(models_dir, f"model_{i}.onnx")
            if os.path.exists(model_path):
                try:
                    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                    self.sessions.append(sess)
                    print(f"Loaded ensemble session: {model_path}")
                except Exception as e:
                    print(f"Error loading ONNX model {model_path}: {e}")
                    
        # Fallback to default single model if no seed-specific models exist
        if len(self.sessions) == 0:
            default_path = os.path.join(models_dir, "model.onnx")
            if os.path.exists(default_path):
                try:
                    sess = ort.InferenceSession(default_path, providers=['CPUExecutionProvider'])
                    self.sessions.append(sess)
                    print(f"Fallback: Loaded single model session from {default_path}")
                except Exception as e:
                    print(f"Error loading default ONNX model {default_path}: {e}")

    def is_ensemble(self):
        """Returns True if multiple sessions are active, False if single model fallback."""
        return len(self.sessions) > 1

    def predict_ensemble(self, batch_img: np.ndarray):
        """
        Runs batch input through all active ONNX sessions, computing mean probabilities,
        mean logits, and uncertainty standard deviation.
        Returns:
            mean_prob: float (or array of floats)
            uncertainty: float or None (if single model fallback)
            mean_logits: np.ndarray
            cls_or_fm: np.ndarray (the second outputs from ONNX like cls_token or feature_map)
        """
        if len(self.sessions) == 0:
            raise RuntimeError("No ONNX inference sessions loaded.")

        logits_list = []
        secondary_outputs_list = []
        probs_list = []

        # Run inference across all sessions
        for sess in self.sessions:
            input_name = sess.get_inputs()[0].name
            outputs = sess.run(None, {input_name: batch_img})
            
            # outputs[0] is logits, outputs[1] is cls_token or feature_map
            logits = outputs[0]
            sec_out = outputs[1]
            
            logits_list.append(logits)
            secondary_outputs_list.append(sec_out)
            
            # Compute sigmoid probability for each model prediction
            prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))
            probs_list.append(prob)

        # Average logits and secondary outputs across the ensemble
        mean_logits = np.mean(logits_list, axis=0)
        mean_sec_out = np.mean(secondary_outputs_list, axis=0)
        
        # Calculate ensemble metrics
        mean_prob = float(np.mean(probs_list))
        
        if len(self.sessions) > 1:
            # Calculate standard deviation as uncertainty metric
            uncertainty = float(np.std(probs_list))
        else:
            # Fallback according to user spec: return None if only 1 model is loaded
            uncertainty = None

        return mean_prob, uncertainty, mean_logits, mean_sec_out
