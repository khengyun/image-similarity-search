import os
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# Load vectors from CSV
def load_vectors_from_csv(collection_name):
    """
    Load vectors, filenames, and classes from a CSV file.
    """
    csv_file_path = f"{collection_name}_vectors.csv"
    logging.info(f"Loading vectors from CSV file: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)

    vectors = df['vector'].apply(lambda x: np.array(eval(x))).tolist()  # Convert string representation of lists back to NumPy arrays
    filenames = df['filename'].tolist()
    classes = df['class'].tolist()

    logging.info(f"Loaded {len(vectors)} vectors from CSV file.")
    return vectors, filenames, classes

# Process an image using pre-loaded vectors
def process_image_with_loaded_vectors(filename, true_label, top_k_values, vectors, filenames, classes, client, collection_name):
    """
    Process image using pre-loaded vectors.
    """
    logging.info(f"Processing image: {filename}")
    metrics = {k: {"precision": 0, "recall": 0, "f1": 0, "map": 0, "correct": False} for k in top_k_values}

    # Find the query vector corresponding to the filename
    if filename in filenames:
        idx = filenames.index(filename)
        query_vector = vectors[idx]
        logging.info(f"Found vector for {filename}")
    else:
        logging.warning(f"Filename {filename} not found in loaded data.")
        return metrics

    # Perform search in Qdrant using the loaded vector
    search_params = qmodels.SearchParams(hnsw_ef=128, exact=False)
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=max(top_k_values),
        search_params=search_params,
        with_payload=["filename", "class"]
    )

    # Collect predicted labels from search results
    predicted_labels = [result.payload["class"] for result in search_results]

    # Compute metrics for each top_k level
    for top_k in top_k_values:
        top_k_predictions = predicted_labels[:top_k]
        true_positives = top_k_predictions.count(true_label)

        # Precision: Number of true positives / number of retrieved results (Top-k)
        metrics[top_k]["precision"] = true_positives / top_k

        # Recall: Number of true positives / total relevant results
        total_relevant = len([label for label in classes if label == true_label])
        metrics[top_k]["recall"] = true_positives / total_relevant if total_relevant > 0 else 0
        logging.info(f"{true_positives}, {total_relevant}")

        # F1-Score: Harmonic mean of Precision and Recall
        precision = metrics[top_k]["precision"]
        recall = metrics[top_k]["recall"]
        metrics[top_k]["f1"] = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # MAP@k (Mean Average Precision at k)
        metrics[top_k]["map"] = map_per_image(true_label, top_k_predictions)

    logging.info(f"Completed processing for image: {filename}")
    return metrics

# MAP@k calculation
def map_per_image(true_label, predicted_labels):
    """
    Compute MAP@k for a single query image.
    """
    score = 0.0
    num_hits = 0.0
    for i, label in enumerate(predicted_labels):
        if label == true_label:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / num_hits if num_hits > 0 else 0.0

# Perform retrieval and evaluate the metrics sequentially
def evaluate_retrieval(client, collection_name, test_set, test_labels):
    logging.info(f"Evaluating retrieval for collection: {collection_name}")
    top_k_values = [5, 10, 15, 20, 25, 30]
    metrics = {k: {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "map": 0} for k in top_k_values}
    map_k_scores = {k: [] for k in top_k_values}
    precision_scores = {k: [] for k in top_k_values}
    recall_scores = {k: [] for k in top_k_values}
    f1_scores = {k: [] for k in top_k_values}
    correct_top_k = {k: 0 for k in top_k_values}

    # Load vectors, filenames, and classes from CSV once per collection
    vectors, filenames, classes = load_vectors_from_csv(collection_name)

    # Process each image sequentially
    total_images = len(test_set)
    for idx, (filename, true_label) in enumerate(zip(test_set, test_labels)):
        logging.info(f"Processing image {idx + 1}/{total_images}: {filename}")
        image_metrics = process_image_with_loaded_vectors(filename, true_label, top_k_values, vectors, filenames, classes, client, collection_name)

        for top_k in top_k_values:
            precision_scores[top_k].append(image_metrics[top_k]["precision"])
            recall_scores[top_k].append(image_metrics[top_k]["recall"])
            map_k_scores[top_k].append(image_metrics[top_k]["map"])
            f1_scores[top_k].append(image_metrics[top_k]["f1"])
            if image_metrics[top_k]["correct"]:
                correct_top_k[top_k] += 1

    # Calculate the final metrics for each top_k
    for top_k in top_k_values:
        metrics[top_k]["accuracy"] = correct_top_k[top_k] / len(test_set)
        metrics[top_k]["precision"] = np.mean(precision_scores[top_k])
        metrics[top_k]["recall"] = np.mean(recall_scores[top_k])
        metrics[top_k]["map"] = np.mean(map_k_scores[top_k])
        metrics[top_k]["f1"] = np.mean(f1_scores[top_k])

        logging.info(f"Collection: {collection_name} - Top-{top_k} Accuracy: {metrics[top_k]['accuracy']:.4f}, "
                     f"MAP@{top_k}: {metrics[top_k]['map']:.4f}, "
                     f"Precision: {metrics[top_k]['precision']:.4f}, "
                     f"Recall: {metrics[top_k]['recall']:.4f}, "
                     f"F1-Score: {metrics[top_k]['f1']:.4f}")
        
    return metrics

# Split dataset
def split_dataset(image_dir, test_size=0.2):
    logging.info(f"Splitting dataset from {image_dir}")
    image_paths = []
    class_labels = []
    for dirpath, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.endswith(".jpg"):
                filepath = os.path.join(dirpath, filename)
                class_name = filepath.split("/")[-2]
                image_paths.append(filepath)
                class_labels.append(class_name)
    train_set, test_set, train_labels, test_labels = train_test_split(image_paths, class_labels, test_size=test_size, stratify=class_labels)
    logging.info(f"Split dataset into {len(train_set)} training images and {len(test_set)} test images.")
    return image_paths, class_labels
    # return train_set, test_set, train_labels, test_labels

# Main function to load data, save vectors, and evaluate metrics
if __name__ == "__main__":
    # Define image directory and test size
    image_directory = "./train/images"
    test_size = 0.2  # 20% for testing

    # Split dataset
    test_set, test_labels = split_dataset(image_directory, test_size=test_size)
    # train_set, test_set, train_labels, test_labels = split_dataset(image_directory, test_size=test_size)

    # Load data from Qdrant, save to CSV, and evaluate retrieval
    results_df = pd.DataFrame(columns=["collection", "top_k", "accuracy", "precision", "recall", "f1", "map"])

    collections = client.get_collections().collections
    for collection in collections:
        collection_name = collection.name
        logging.info(f"Starting evaluation for collection: {collection_name}")
        metrics = evaluate_retrieval(client, collection_name, test_set, test_labels)

        # Save results to CSV
        csv_file_path = "retrieval_results.csv"
        for top_k, metric in metrics.items():
            new_row = pd.DataFrame([{
                "collection": collection_name,
                "top_k": top_k,
                "accuracy": metric["accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "f1": metric["f1"],
                "map": metric["map"]
            }])
            new_row = new_row.dropna(axis=1, how='all')
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_csv(csv_file_path, index=False)

    logging.info(f"Results appended to {csv_file_path}")
