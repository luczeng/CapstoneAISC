import torch
from pathlib import Path
from Capstone.data.datasets import DatasetRSNA_jpg
from torch.utils.data import DataLoader
import torch.nn as nn


def evaluate_model(model, image_folder_path, label_file_path, mini_batch_size, net_type):
    """
        Evaluates the model on the given dataset

        :param model model to evaluate
        :param image_folder_path path to folder containing dicom images
        :param label_file_path path to .csv file that should contain labels
        :return accuracy
    """

    if Path(label_file_path).suffix != ".csv":
        raise ValueError("Label file should be of csv format")

    dataset = DatasetRSNA_jpg(net_type, image_folder_path, label_file_path)
    dataloader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=False)

    error = 0
    softmax = nn.Softmax(dim=1)
    for batch in dataloader:

        inferences = model(batch["image"])
        prob = softmax(inferences).detach().cpu().numpy()[0]
        print("{:.2f} {:2f}".format(prob[0], prob[1]))
        inferences = torch.argmax(inferences, axis=1)
        error += torch.sum(torch.abs(batch["label"] - inferences))

    return error
