import torch
from pathlib import Path
from Capstone.data.datasets import DatasetRSNA_jpg
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support


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
    y_pred = []
    y_true = []
    for batch in dataloader:

        inferences = torch.argmax(model(batch["image"]), axis=1)
        y_pred.append(inferences.item())
        y_true.append(batch["label"].item())

    pr, rc, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    pr_micro, rc_micro, f_score_micro, support = precision_recall_fscore_support(y_true, y_pred, average="micro")
    pr_macro, rc_macro, f_score_macro, support = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print('\nPrecision, recall and fscore:')
    print(f'Per cat: [{pr[0]:.2f} {pr[1]:.2f}], [{rc[0]:.2f} {rc[1]:.2f}], [{f_score[0]:.2f} {f_score[1]:.2f}]')
    print(f'Micro  : {pr_micro:.2f}, {rc_micro:.2f}, {f_score_micro:.2f}')
    print(f'macro  : {pr_macro:.2f}, {rc_macro:.2f}, {f_score_macro:.2f}')

    return error
