import torch
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np

class TransformerFeatureExtractor:
    def __init__(self, model_name):
        """
        Initialize the model and processor based on the model type.
        
        Args:
            model_name (str): The HuggingFace model name (e.g., "google/vit-base-patch16-224", "Mamba-Codestral-7B-v0.1").
        """
        self.model_name = model_name

        # Initialize different models based on the type
        if model_name.startswith("vit"):
            if model_name == "vit_base_patch16_224":
                self.model_name = "google/vit-base-patch16-224"
            elif model_name == "vit_large_patch16_224":
                self.model_name = "google/vit-large-patch16-224"
            elif model_name == "vit_huge_patch14_224":
                self.model_name = "google/vit-huge-patch14-224-in21k"
            elif model_name == "facebook/dino-vitb16":
                self.model_name = "facebook/dino-vitb16"
            
            self.model = ViTModel.from_pretrained(self.model_name, force_download=True)
            self.image_processor = ViTImageProcessor.from_pretrained(self.model_name)

            # Extract the output size from the model's hidden layer
            self.output_size = self.model.config.hidden_size
            self.model_type = 'vit'

        elif model_name == "Mamba-Codestral-7B-v0.1":
            # Initialize the Mamba text model
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", force_download=True)
            self.model = AutoModel.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1", force_download=True)

            # Extract the output size from the model's hidden layer for text-based models
            self.output_size = self.model.config.hidden_size
            self.model_type = 'mamba'

        else:
            raise ValueError(f"Model {model_name} is not supported.")

        self.model.eval()

    def extract_vit_features(self, image_path):
        """
        Extract image features using a Vision Transformer (ViT) model.
        
        Args:
            image_path (str): Path to the image file.
        
        Returns:
            np.ndarray: The extracted image feature vector.
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

    def extract_mamba_features(self, text):
        """
        Extract text features using the Mamba model.
        
        Args:
            text (str): Input text for feature extraction.
        
        Returns:
            np.ndarray: The extracted text feature vector.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Ensure model and inputs are on the same device (e.g., CUDA if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Extract features with no gradient computation
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract the last hidden state or other relevant feature (depending on your use case)
        feature_vector = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        # Normalize the feature vector to unit length (L2 normalization)
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()

    def extract(self, input_data):
        """
        General extract method that delegates to the appropriate feature extractor based on the model type.

        Args:
            input_data (str): Path to the image file or text input, depending on the model.
        
        Returns:
            np.ndarray: The extracted feature vector.
        """
        if self.model_type == 'vit':
            return self.extract_vit_features(input_data)
        elif self.model_type == 'mamba':
            return self.extract_mamba_features(input_data)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
