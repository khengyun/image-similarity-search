import torch
from transformers import ViTModel, ViTImageProcessor, MambaModel, MambaForCausalLM
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np

class TransformerFeatureExtractor:
    def __init__(self, model_name):
        """
        Initialize the Vision Transformer (ViT) model and image processor.

        Args:
            model_name (str): The HuggingFace model name (e.g., "google/vit-base-patch16-224", "google/vit-large-patch16-224").
        """
        
        self.model_name = model_name

        if model_name == "vit_base_patch16_224":
            self.model_name = "google/vit-base-patch16-224"
            self.model = MambaModel.from_pretrained(self.model_name, token=True)
        else:
            raise ValueError(f"Model {model_name} is not supported.")
        
        self.model.eval()  
        self.image_processor = MambaForCausalLM.from_pretrained(self.model_name)  

        # Extract the output size from the model (CLS token size or equivalent for the model)
        self.output_size = self.model.config.hidden_size  

    def extract(self, image_path):
        """
        Extract features from an image using the Vision Transformer (ViT).

        Args:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: The extracted feature vector.
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")

        # Ensure model and inputs are on the same device (e.g., CUDA if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract features with no gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the pooled output (CLS token output), which is typically used as the image feature
        feature_vector = outputs.pooler_output.squeeze().cpu().numpy()

        # Normalize the feature vector to unit length (L2 normalization)
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()
