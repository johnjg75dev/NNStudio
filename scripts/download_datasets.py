"""
scripts/download_datasets.py
Manual script to download predefined datasets (e.g. MNIST) and update DB.
"""
import os
import sys
import json
import gzip
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app, db
from app.models import Dataset

DATASETS_DIR = Path("instance/datasets")

def download_mnist():
    """Download MNIST data and save to instance/datasets/mnist."""
    mnist_dir = DATASETS_DIR / "mnist"
    mnist_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    print("Downloading MNIST dataset...")
    for f in files:
        target = mnist_dir / f
        if target.exists():
            print(f"  {f} already exists, skipping.")
            continue

        print(f"  Downloading {f}...")
        try:
            urllib.request.urlretrieve(base_url + f, target)
        except Exception as e:
            print(f"  Error downloading {f}: {e}")
            return False

    return True

def update_database():
    """Update all predefined MNIST datasets in DB to mark as downloaded."""
    app = create_app()
    with app.app_context():
        datasets = Dataset.query.filter_by(ds_type="mnist", is_predefined=True).all()
        if not datasets:
            print("No predefined MNIST datasets found in database.")
            return

        for ds in datasets:
            ds.downloaded = True
            ds.file_path = "instance/datasets/mnist"
            print(f"Updated dataset '{ds.name}' (ID: {ds.id}) for user {ds.user_id}")

        db.session.commit()
        print(f"Successfully updated {len(datasets)} dataset records.")

if __name__ == "__main__":
    if download_mnist():
        update_database()
    else:
        print("Download failed.")
