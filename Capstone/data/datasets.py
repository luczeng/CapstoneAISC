from torch.utils.data import Dataset
import pydicom
import csv
import torch


class DatasetRSNA(Dataset):
    def __init__(self, cfg):
        """
            :param cfg parsed config
        """

        self.cfg = cfg
        with open(cfg.train_label_path, newline="") as f:
            reader = csv.reader(f)
            self.data = list(reader)

    def __getitem__(self, idx):
        """
            Parses the DICOM file and return image and corresponding label

            :param idx
        """

        img_info = self.data[idx]

        img_path = self.cfg.train_dataset_path + "/" + img_info[0] + ".dcm"
        dataset = pydicom.dcmread(img_path)
        image = dataset.pixel_array
        image = torch.tensor(image[None, :, :]).type(torch.float)

        label = int(img_info[5])

        sample = {"image": image, "label": label}

        return sample

    def __len__(self):

        return len(self.data) - 1
