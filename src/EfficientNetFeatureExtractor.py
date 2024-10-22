from keras.api.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from keras.api.applications.xception import Xception
from keras.api.applications.vgg16 import VGG16
from keras.api.applications.vgg19 import VGG19

from keras.api.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from keras.api.applications.xception import preprocess_input as preprocess_input_xception
from keras.api.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.api.applications.vgg19 import preprocess_input as preprocess_input_vgg19

from keras.api.preprocessing import image as keras_image
from keras.api.models import Model
import numpy as np

class EfficientNetFeatureExtractor:
    def __init__(self, model_name):
        if model_name == "efficientnet_b0":
            base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b1":
            base_model = EfficientNetB1(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b2":
            base_model = EfficientNetB2(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b3":
            base_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b4":
            base_model = EfficientNetB4(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b5":
            base_model = EfficientNetB5(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b6":
            base_model = EfficientNetB6(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "efficientnet_b7":
            base_model = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_efficientnet
        elif model_name == "vgg16":
            base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_vgg16
        elif model_name == "vgg19":
            base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_vgg19
        elif model_name == "xception":
            base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess_input = preprocess_input_xception
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        # Store the model that will be used for feature extraction
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

        # Extract the output size from the model (should be the number of dimensions in the final vector)
        self.output_size = self.model.output_shape[-1]  # The size of the output vector

    def extract(self, image_path):
        # Load image and preprocess for the chosen model
        img = keras_image.load_img(image_path, target_size=(224, 224))  # Resize to 224x224 for consistency
        img_data = keras_image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = self.preprocess_input(img_data)

        # Extract features
        features = self.model.predict(img_data)
        return features.flatten()
