import torch
from pathlib import Path
from Capstone.data.datasets import DatasetRSNA_jpg
from torch.utils.data import DataLoader


def evaluate_model(model, image_folder_path, label_file_path, cfg, net_type):
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
    dataloader = DataLoader(dataset, batch_size=cfg.mini_batch_size, shuffle=False)

    error = 0
    for batch in dataloader:

        inferences = torch.argmax(model(batch["image"]), axis=1)
        error += torch.sum(torch.abs(batch["label"] - inferences))

    return error
