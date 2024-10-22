import torch
from torch import nn
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import asyncio
from concurrent.futures import ThreadPoolExecutor

class FeatureExtractor:
    def __init__(self, modelname):
        # Load the model dynamically based on model name
        self.model = timm.create_model(modelname, pretrained=True, num_classes=0, global_pool="avg")
        self.model.eval()

        # Get the number of features output by the model dynamically
        self.num_features = self.model.num_features

        # Create a fully connected layer that maps the model's output to 128 dimensions
        self.fc = nn.Linear(self.num_features, 128)
        self.fc.eval()

        # Create preprocessing pipeline based on the model's input requirements
        config = resolve_data_config({}, model=self.model)
        self.preprocess = create_transform(**config)

    def extract(self, imagepath):
        # Load and preprocess the image
        input_image = Image.open(imagepath).convert("RGB")
        input_image = self.preprocess(input_image)

        # Convert image to tensor and add batch dimension
        input_tensor = input_image.unsqueeze(0)

        with torch.no_grad():
            # Extract features using the base model
            output = self.model(input_tensor)

            # Pass the output through the fully connected layer to reduce to 128 dimensions
            output_128d = self.fc(output)

        # Normalize the feature vector to unit length (L2 normalization)
        feature_vector = output_128d.squeeze().numpy()
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


async def search_images_async(client, query_vector, collection_name, hnsw_ef=128):
    search_params = qmodels.SearchParams(
        hnsw_ef=hnsw_ef,
        exact=False
    )
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=10,
        search_params=search_params,
        with_payload=["filename", "class"]
    )


async def load_image_async(filename):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        return await loop.run_in_executor(executor, Image.open, filename)


async def display_images(results):
    images = []
    tasks = [load_image_async(result.payload["filename"]) for result in results]
    loaded_images = await asyncio.gather(*tasks)

    for idx, img in enumerate(loaded_images):
        img = img.resize((150, 150))
        images.append(img)

    width, height = 150 * 5, 150 * 2
    concatenated_image = Image.new("RGB", (width, height))

    for idx, img in enumerate(images):
        x, y = idx % 5, idx // 5
        concatenated_image.paste(img, (x * 150, y * 150))

    concatenated_image.show()


async def perform_search_and_display(client, query_image, backbones):
    extractor_results = {}

    for backbone in backbones:
        # Initialize the feature extractor for each backbone
        extractor = FeatureExtractor(backbone)

        # Extract features from the query image
        query_vector = extractor.extract(query_image)

        # Perform asynchronous search for the current backbone
        collection_name = f"image_embeddings_{backbone}"
        results = await search_images_async(client, query_vector, collection_name)

        # Store the results in a dictionary for later use/display
        extractor_results[backbone] = results

    # Display results from each backbone
    for backbone, results in extractor_results.items():
        print(f"Results for {backbone}:")
        await display_images(results)


# # Main function to run the setup
# async def main():
#     # Initialize the Qdrant client
#     client = QdrantClient(host="localhost", port=6333)

#     # Define the query image path
#     query_image = "train/images/tiramisu/2680.jpg"

#     # Define the backbones you want to use
#     backbones = ["resnet34", "efficientnet_b0", "mobilenetv2_100"]

#     # Perform search and display for each backbone
#     await perform_search_and_display(client, query_image, backbones)

# # Run the asynchronous main function
# asyncio.run(main())
