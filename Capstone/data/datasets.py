from torch.utils.data import Dataset
import pydicom
import csv
import torch


class DatasetRSNA(Dataset):
    def __init__(self, net_type, image_folder_path, label_file_path):
        """
        """

        with open(label_file_path, newline="") as f:
            reader = csv.reader(f)
            self.data = list(reader)

        self.net_type = net_type
        self.image_folder_path = image_folder_path
        self.label_file_path = label_file_path

    def __getitem__(self, idx):
        """
            Parses the DICOM file and return image and corresponding label

            :param idx
        """

        img_info = self.data[idx]

        img_path = self.image_folder_path + "/" + img_info[0] + ".dcm"
        dataset = pydicom.dcmread(img_path)
        image = dataset.pixel_array
        image = torch.tensor(image[None, :, :]).type(self.net_type)

        label = torch.tensor(int(img_info[5])).type(torch.cuda.LongTensor)

        sample = {"image": image, "label": label}

        return sample

    def __len__(self):

        return len(self.data) - 1
