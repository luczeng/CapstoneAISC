from pathlib import Path
import pydicom
import numpy as np


def load_dicom(img_path :str):
    '''
        Loads dicom image into numpy array

        :param img_path to .dcm iamge
        :return img: loaded image
    '''

    data = dataset = pydicom.dcmread(img_path)
    img = data.pixel_array

    return img


def load_folder(folder_path :str):
    '''
        Loads image within a numpy array

        :param folder_path path to folder containing only .dcm images
    '''

    fp = Path(folder_path)
    imgs = np.zeros((len(list(fp.glob('*'))), 1, 1024, 1024 ))
    for i, f in enumerate(Path(fp).iterdir()):
        imgs[i, :, :, :] = load_dicom(str(f))[None, :, :]

    return imgs
