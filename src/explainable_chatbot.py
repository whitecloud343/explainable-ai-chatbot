import os
import torch
import numpy as np
import shap
import lime.lime_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gradcam.utils import visualize_cam
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import logging

class ExplainableChatbot:
    """
    A chatbot with explainable AI capabilities using SHAP, LIME, and GradCAM.
    Provides transparency in decision-making for customer service or educational applications.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        # Initialize explainers
        self.shap_explainer = shap.Explainer(self._model_predict)
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=self._get_class_names()
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_class_names(self) -> List[str]:
        """Get class names based on the number of labels."""
        if self.num_labels == 3:
            return ["Negative", "Neutral", "Positive"]
        return [f"Class_{i}" for i in range(self.num_labels)]
    
    def _model_predict(self, texts: List[str]) -> np.ndarray:
        """Model prediction function for SHAP."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.cpu().numpy()
    
    def explain_with_shap(self, text: str) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the model's decision.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary containing SHAP values and visualization data
        """
        self.logger.info("Generating SHAP explanation...")
        
        # Generate SHAP values
        shap_values = self.shap_explainer([text])
        
        # Create visualization
        plt.figure()
        shap.plots.text(shap_values[0])
        plt.tight_layout()
        
        # Save visualization
        plt.savefig("shap_explanation.png")
        plt.close()
        
        return {
            "shap_values": shap_values.values,
            "base_values": shap_values.base_values,
            "visualization_path": "shap_explanation.png"
        }
    
    def explain_with_lime(self, text: str) -> Dict[str, Any]:
        """
        Generate LIME explanations for the model's decision.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary containing LIME explanation data
        """
        self.logger.info("Generating LIME explanation...")
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            text,
            self._model_predict,
            num_features=10,
            num_samples=100
        )
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        exp.as_pyplot_figure()
        plt.tight_layout()
        
        # Save visualization
        plt.savefig("lime_explanation.png")
        plt.close()
        
        return {
            "local_exp": exp.local_exp,
            "score": exp.score,
            "visualization_path": "lime_explanation.png"
        }
    
    def explain_with_gradcam(self, text: str) -> Dict[str, Any]:
        """
        Generate GradCAM explanations for the model's attention.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary containing GradCAM visualization data
        """
        self.logger.info("Generating GradCAM explanation...")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get model outputs and gradients
        self.model.zero_grad()
        output = self.model(**inputs)
        output.logits.backward()
        
        # Get attention weights
        attention = self.model.base_model.encoder.layer[-1].attention.self.get_attention_map()
        
        # Create heatmap visualization
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention.detach().cpu().numpy()[0],
            cmap="YlOrRd",
            xticklabels=self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]),
            yticklabels=self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        )
        plt.title("Attention Heatmap")
        plt.tight_layout()
        
        # Save visualization
        plt.savefig("gradcam_explanation.png")
        plt.close()
        
        return {
            "attention_weights": attention.detach().cpu().numpy(),
            "visualization_path": "gradcam_explanation.png"
        }
    
    def get_complete_explanation(self, text: str) -> Dict[str, Any]:
        """
        Generate comprehensive explanations using all methods.
        
        Args:
            text: Input text to explain
            
        Returns:
            Dictionary containing all explanations and metadata
        """
        self.logger.info(f"Generating complete explanation for text: {text}")
        
        # Get model prediction
        prediction = self._model_predict([text])[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Generate explanations
        shap_exp = self.explain_with_shap(text)
        lime_exp = self.explain_with_lime(text)
        gradcam_exp = self.explain_with_gradcam(text)
        
        return {
            "input_text": text,
            "prediction": {
                "class": self._get_class_names()[predicted_class],
                "confidence": float(confidence)
            },
            "explanations": {
                "shap": shap_exp,
                "lime": lime_exp,
                "gradcam": gradcam_exp
            },
            "model_info": {
                "model_name": self.model_name,
                "num_labels": self.num_labels
            }
        }
    
    def save_explanations(self, explanation_data: Dict[str, Any], output_dir: str = "explanations"):
        """
        Save all explanations and visualizations to disk.
        
        Args:
            explanation_data: Dictionary containing explanation data
            output_dir: Directory to save explanations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        import json
        with open(os.path.join(output_dir, "explanation_metadata.json"), "w") as f:
            json.dump({
                "input_text": explanation_data["input_text"],
                "prediction": explanation_data["prediction"],
                "model_info": explanation_data["model_info"]
            }, f, indent=2)
        
        # Save visualizations
        for method, exp_data in explanation_data["explanations"].items():
            if "visualization_path" in exp_data:
                src_path = exp_data["visualization_path"]
                if os.path.exists(src_path):
                    dst_path = os.path.join(output_dir, f"{method}_explanation.png")
                    os.rename(src_path, dst_path)
                    self.logger.info(f"Saved {method} visualization to {dst_path}")

def main():
    """Example usage of ExplainableChatbot."""
    # Initialize chatbot
    chatbot = ExplainableChatbot()
    
    # Example text
    text = "I really enjoyed the customer service experience. The representative was very helpful."
    
    # Get explanations
    explanations = chatbot.get_complete_explanation(text)
    
    # Save explanations
    chatbot.save_explanations(explanations)
    
    print(f"Prediction: {explanations['prediction']}")
    print(f"Explanations saved to 'explanations' directory")

if __name__ == "__main__":
    main()