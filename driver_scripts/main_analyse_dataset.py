import argparse
from pathlib import Path
import csv


def parse_args():

    parser = argparse.ArgumentParser(description="Scripts that allows to quickly see the nber of positives and negatives")
    parser.add_argument("-l", "--label_file_path", type=str, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    if Path(args.label_file_path).suffix != ".csv":
        raise ValueError("Label file should be of csv format")

    with open(args.label_file_path, newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    nb_positives = 0
    nb_negatives = 0
    for sample in data:
        if int(sample[5]) == 0:
            nb_negatives += 1
        else:
            nb_positives += 1

    print(nb_positives, nb_negatives)
