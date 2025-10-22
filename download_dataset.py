import kagglehub
import os
os.environ['KAGGLEHUB_CACHE'] = "./dataset/"
# Download latest version
path = kagglehub.dataset_download("balraj98/deepglobe-road-extraction-dataset")

print("Path to dataset files:", path)