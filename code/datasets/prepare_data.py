import gzip
import os
import struct
import urllib.request

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "mnist")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

URLS = {
    "train-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_raw_files():
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in URLS.items():
        path = os.path.join(RAW_DIR, name)
        if not os.path.exists(path):
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"{name} already present")


def read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_raw_data():
    images = np.concatenate(
        [
            read_images(os.path.join(RAW_DIR, "train-images-idx3-ubyte.gz")),
            read_images(os.path.join(RAW_DIR, "t10k-images-idx3-ubyte.gz")),
        ]
    ).reshape(-1, 28 * 28)
    labels = np.concatenate(
        [
            read_labels(os.path.join(RAW_DIR, "train-labels-idx1-ubyte.gz")),
            read_labels(os.path.join(RAW_DIR, "t10k-labels-idx1-ubyte.gz")),
        ]
    ).astype(np.int64)
    return images, labels


def clean_data(images, labels):
    x = images.astype(np.float32)

    # Check for missing values and impute them with zero
    n_missing = int(np.isnan(x).sum())
    x = np.nan_to_num(x, nan=0.0)

    # Remove blank images and outliers by pixel sum (3 sigma rule)
    sums = x.sum(axis=1)
    low = sums.mean() - 3 * sums.std()
    high = sums.mean() + 3 * sums.std()
    mask = (sums >= low) & (sums <= high) & (sums > 0)
    n_outliers = int((~mask).sum())
    x = x[mask]
    y = labels[mask]
    print(f"missing values: {n_missing}")
    print(f"outliers removed: {n_outliers}")
    return x.astype(np.uint8), y


def split_data(images, labels):
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(images))
    n_test = int(round(len(images) * 0.1))
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.savez(os.path.join(PROCESSED_DIR, "train.npz"), images=images[train_idx], labels=labels[train_idx])
    np.savez(os.path.join(PROCESSED_DIR, "test.npz"), images=images[test_idx], labels=labels[test_idx])
    print(f"train samples: {len(train_idx)}")
    print(f"test samples: {len(test_idx)}")


def main():
    download_raw_files()
    images, labels = load_raw_data()
    print(f"raw samples: {len(images)}")
    images, labels = clean_data(images, labels)
    split_data(images, labels)
    print("done")


if __name__ == "__main__":
    main()
