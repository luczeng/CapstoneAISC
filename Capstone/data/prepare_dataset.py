# Splits the train set into a test and train set
import csv
import os
import shutil
from pathlib import Path


def read_csv_to_list(csv_path):
    """
        Read csv to list

        :param csv_path
    """

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


def save_labels_to_disk(labels: list, label_path: str):
    """
        Saves labels to csv

        :param labels
        :param label_path
    """

    with open(label_path, "w") as result_file:
        wr = csv.writer(result_file, dialect="excel")
        wr.writerows(labels)


def split_test_train(train_folder_path, train_labels, test_folder, n_test_images):
    """
        Splits the train set into test images

        :param train_folder_path
        :param train_labels
        :param test_folder
        :param n_test_images number of test_images
        :return train_labels
        :return test_labels
    """

    os.makedirs(test_folder, exist_ok=True)

    data = read_csv_to_list(train_labels)
    # Prepare test labels and move images to new folder
    labels = []
    for img in data[1:n_test_images]:
        # Input and new image paths
        # print(type(train_folder_path),type(img[0]))
        img_path = train_folder_path / (img[0] + ".dcm")
        new_img_path = test_folder / (img[0] + ".dcm")
        if Path(img_path).exists():  # there can be several annotations per image
            shutil.move(img_path, new_img_path)
            labels.append(img)

    # Prepare train labels. Removes duplicate as we dont need them.
    train_labels = []
    img_list_names = []
    for idx, label in enumerate(data[n_test_images + 1 :]):
        if (label[0] in img_list_names) and (idx != 0):
            continue
        img_list_names.append(label[0])
        train_labels.append(label)

    # labels.insert(0, data[0])
    # train_labels.insert(0, data[0])
    return train_labels, labels
