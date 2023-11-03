import os
import shutil
import random
import argparse

# dataset_path = "/content/cnndataset"
# train_ratio = 0.8


def split_dataset(dataset_path, train_ratio, dis_path):
    """
    Split a dataset into training and validation sets.

    Parameters:
    - dataset_path (str): The path to the source dataset.
    - train_ratio (float): The ratio of data to be allocated to the training set (e.g., 0.8 for an 80-20 split).
    - dis_path (str): The path to the directory where the split data will be saved.

    """
    train = dis_path + '/train'
    val = dis_path + '/val'
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        files = os.listdir(class_path)
        random.shuffle(files)

        split_index = int(train_ratio * len(files))
        train_files = files[:split_index]
        val_files = files[split_index:]

        train_folder = os.path.join(train, class_folder)
        val_folder = os.path.join(val, class_folder)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_folder, file)
            shutil.copy(src, dst)

        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(val_folder, file)
            shutil.copy(src, dst)


def parse_args():
    parser = argparse.ArgumentParser(description="rotate and skew currection")
    parser.add_argument("--dataset_path", required=True, help="Path to the input scaned document image")
    parser.add_argument("--train_ratio", required=True, type=float, help="the ratio of split dataset")
    parser.add_argument("--dis_path", required=True, help="Path to the output scaned document image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.dataset_path, args.train_ratio, args.dis_path)
