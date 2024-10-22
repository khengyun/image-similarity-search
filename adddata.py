import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from src.FeatureExtractor import FeatureExtractor
from src.EfficientNetFeatureExtractor import EfficientNetFeatureExtractor
from src.TransformerFeatureExtractor import TransformerFeatureExtractor


# Connect to Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# Function to ensure the collection is created before any operation
def ensure_collection_exists(collection_name, vector_size=128):
    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        print(f"Collection '{collection_name}' does not exist. Creating now.")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

from collections import defaultdict

# Function to create and insert into collections for each model
def create_and_insert_into_collection(model_name, collection_suffix, image_dir, class_image_limit=10):
    """
    Create and insert data into a Qdrant collection for the given model.

    Args:
        model_name (str): Name of the model to use for feature extraction.
        collection_suffix (str): Suffix to be used in the Qdrant collection name.
        image_dir (str): Path to the directory containing images.
        class_image_limit (int): Maximum number of images to process per class.
    """


    if "resnet" in model_name.lower():
        extractor = FeatureExtractor(model_name)
        vector_size = 128  # Default vector size for other models
        
    if "vit" in model_name.lower() or "mamba" in model_name.lower():
        extractor = TransformerFeatureExtractor(model_name)
        vector_size = extractor.output_size  
        
    else:
        extractor = EfficientNetFeatureExtractor(model_name)
        vector_size = extractor.output_size  


    # Create a unique collection name based on the model name
    collection_name = f"image_embeddings_{collection_suffix}"

    # Ensure the collection exists before attempting to insert data
    ensure_collection_exists(collection_name, vector_size=vector_size)

    # Dictionary to keep track of the number of images processed per class
    class_count = defaultdict(int)

    # Insert data into the collection
    point_id = 0
    total_limit = 0
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                filepath = os.path.join(dirpath, filename)

                # Extract the class name from the file path
                class_name = filepath.split("/")[-2]

                # Check if the class has reached the image limit
                # if class_count[class_name] >= class_image_limit:
                #     continue  # Skip this image if the class limit is reached

                # Extract image embedding
                image_embedding = extractor.extract(filepath)

                # Insert into Qdrant collection with generated ID
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            "id": point_id,  # Unique ID for each vector
                            "vector": image_embedding,
                            "payload": {
                                "filename": filepath,
                                "class": class_name  # Save the class name
                            }
                        }
                    ]
                )

                # Update counters
                class_count[class_name] += 1
                point_id += 1

                # Optional: stop processing if a total image limit is reached (commented out)
                # if point_id >= total_limit:
                #     break

    # Optional: Create an index for more efficient searching
    try:
        client.create_index(collection_name=collection_name, field_name="vector")
        print(f"Index created for collection: {collection_name}")
    except Exception as e:
        print(f"Index already exists or failed to create index for {collection_name}: {str(e)}")


# Main logic to insert data into different collections for different models
if __name__ == "__main__":
    # Define the image directory and the models you want to use
    image_directory = "./train/images"
    models = [
        {"name": "resnet18", "suffix": "resnet18"},
        {"name": "resnet26", "suffix": "resnet26"},
        {"name": "resnet34", "suffix": "resnet34"},
        {"name": "resnet50", "suffix": "resnet50"},
        {"name": "resnet101", "suffix": "resnet101"},
        {"name": "resnet152", "suffix": "resnet152"},
        
        {"name": "efficientnet_b0", "suffix": "efficientnet_b0"},
        {"name": "efficientnet_b1", "suffix": "efficientnet_b1"},
        {"name": "efficientnet_b2", "suffix": "efficientnet_b2"},
        {"name": "efficientnet_b3", "suffix": "efficientnet_b3"},
        {"name": "efficientnet_b4", "suffix": "efficientnet_b4"},
        {"name": "efficientnet_b5", "suffix": "efficientnet_b5"},
        {"name": "efficientnet_b6", "suffix": "efficientnet_b6"},
        {"name": "efficientnet_b7", "suffix": "efficientnet_b7"},
                
        {"name": "xception", "suffix": "xception"},
        {"name": "vgg16", "suffix": "vgg16"},
        {"name": "vgg19", "suffix": "vgg19"},
        
        {"name": "vit_base_patch16_224", "suffix": "vit_base_patch16_224"},
        {"name": "vit_large_patch16_224", "suffix": "vit_large_patch16_224"},
        {"name": "vit_huge_patch14_224", "suffix": "vit_huge_patch14_224"},
        
        
        
        
        
    ]

    # Loop over each model and insert data into its respective collection
    for model in models:
        create_and_insert_into_collection(
            model_name=model["name"],
            collection_suffix=model["suffix"],
            image_dir=image_directory,
            class_image_limit=5
        )
