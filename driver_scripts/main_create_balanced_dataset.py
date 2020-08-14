from pathlib import Path
from Capstone.data.prepare_dataset import save_labels_to_disk
import csv


root_folder = "dataset/rsna-pneumonia-detection-challenge"

labels = 'dataset/rsna-pneumonia-detection-challenge/cleaned_train_labels.csv'
balanced_labels = 'dataset/rsna-pneumonia-detection-challenge/balanced_labels.csv'


if __name__ == "__main__":

    with open(labels, newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    train_labels = []
    nb_positives = 0
    for label in data:
        # If we have a positive sample, we activate the flag
        if int(label[5]) == 1:
            train_labels.append(label)
            nb_positives += 1

    negative_labels = 0
    for label in data:
        if int(label[5]) == 0:
            if negative_labels <= nb_positives:
                train_labels.append(label)
                negative_labels += 1

    save_labels_to_disk(train_labels, balanced_labels)
