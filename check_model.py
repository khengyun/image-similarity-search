from qdrant_client import QdrantClient

# Connect to Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# Retrieve all collections
collections = client.get_collections()

# Display the collections
print("Available collections:")
for collection in collections.collections:
    print(collection.name)
