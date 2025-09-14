"""
Data preparation script for gym exercises recognition.
Downloads and preprocesses data from UCI ML Repository.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

def download_gym_data():
    """
    Download gym exercises data from UCI ML Repository.
    
    Note: You'll need to find the correct dataset ID for your specific
    gym exercises dataset. Use list_available_datasets() to browse.
    """
    
    # Example - replace with actual dataset ID
    # dataset_id = 123  # Replace with actual UCI dataset ID
    
    # Uncomment and modify when you know the dataset ID:
    # dataset = fetch_ucirepo(id=dataset_id)
    # X = dataset.data.features
    # y = dataset.data.targets
    
    # # Save raw data
    # X.to_csv('../data/raw/features.csv', index=False)
    # y.to_csv('../data/raw/targets.csv', index=False)
    
    print("Data download function ready - update with correct dataset ID")

def list_datasets():
    """List available datasets to find the gym exercises dataset."""
    datasets = list_available_datasets()
    print("Available datasets:")
    for dataset in datasets[:10]:  # Show first 10
        print(f"ID: {dataset['id']}, Name: {dataset['name']}")

if __name__ == "__main__":
    list_datasets()
    download_gym_data()
