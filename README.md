# Image Similarity Search Project

## Introduction

This project is designed to perform image similarity search using vector databases. It involves converting data from images with associated classes into a vector database and calculating retrieval metrics based on the stored data. The project consists of two key Python scripts:

1. `adddata.py` – Converts image data into vector representations and stores them in a vector database (vectordb).
2. `kakulator.py` – Computes retrieval metrics after the data is successfully added to the vector database.

## File Descriptions

### Image Directory Setup

The project requires an image directory where all the input images are stored. This directory should be organized properly, with images grouped by their respective classes (if applicable).

- Ensure the image directory path is correctly specified in `adddata.py` so that the script can access the images.
- The directory can have subfolders for each class, or all images can be kept together depending on the use case.

Example structure:

```
image_directory/
    class_1/
        image1.jpg
        image2.jpg
    class_2/
        image3.jpg
        image4.jpg
```

### `adddata.py`

**Note**: The script requires an image directory, which should be properly set up before running. See the Image Directory Setup section for more details.
This script processes images and converts them into vector embeddings that can be stored in a vector database. It takes image inputs along with their respective classes and:

- Extracts features from each image using pre-trained models or feature extraction techniques.
- Converts these features into vectors.
- Stores the resulting vector representations in a vector database (vectordb), enabling efficient image similarity search.

#### How to Use

1. Make sure you have installed all required dependencies, including any machine learning libraries for image processing.
2. Run `adddata.py` using Python:

   ```sh
   python adddata.py
   ```

3. You may need to configure database connection settings or provide input image files, depending on the implementation.

### `kakulator.py`

This script calculates various metrics to evaluate the retrieval performance of the system once the data is available in the vector database. Some metrics that may be computed include:

- Precision and Recall
- Mean Average Precision (mAP)
- Top-k Accuracy

#### How to Use

1. Ensure that `adddata.py` has successfully added data to the vector database.
2. Run `kakulator.py` to compute the metrics:
   ```sh
   python kakulator.py
   ```
3. The output will provide a detailed evaluation report of the retrieval metrics.

## Dependencies

- Python 3.x
- Required libraries: numpy, pandas, scikit-learn, vector database client (e.g., Pinecone, Weaviate, or similar)

Install the required dependencies using:

```sh
pip install -r requirements.txt
```

## Setup Instructions

### Docker-Compose Setup

If you prefer to run the project using Docker, follow these steps:

1. Ensure Docker and Docker-Compose are installed on your system.
2. Create a `docker-compose.yml` file with the necessary services, such as the vector database and application services.
3. Use the following command to start the services:
   ```sh
   docker compose up --build
   ```
4. The application should now be running, and you can interact with it as described above.

5. Clone the repository.
   ```sh
   git clone https://github.com/khengyun/image-similarity-search.git
   ```
6. Install the required dependencies using `requirements.txt`.
7. Configure any necessary database or environment variables in the project configuration.

## Contribution

If you want to contribute to this project, feel free to fork the repository and create a pull request.
