from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import os
from src.FeatureExtractor import FeatureExtractor

# Connect to Qdrant instance
client = QdrantClient(url="http://localhost:6333")

collection_name = "image_embeddings"

# Check if collection exists, delete if necessary
if client.get_collection(collection_name=collection_name):
    client.delete_collection(collection_name=collection_name)

# Create collection in Qdrant
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
)

# Initialize feature extractor
extractor = FeatureExtractor("resnet34")

root = "./train/images"
insert = True
if insert is True:
    point_id = 0 
    # Loop through images and extract embeddings
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".jpg"):
                filepath = os.path.join(dirpath, filename)
                
                # Extract image embedding
                image_embedding = extractor(filepath)
                
                # Insert into Qdrant collection with generated ID
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        {
                            "id": point_id,  # Unique ID for each vector
                            "vector": image_embedding,
                            "payload": {
                                "filename": filepath,
                                "class": filepath.split("/")[-2]
                            }
                        }
                    ]
                )
                point_id += 1 
                
# Optional: Create an index for more efficient searching
client.create_index(collection_name=collection_name, field_name="vector")
