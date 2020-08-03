import pydicom
from pathlib import Path
from PIL import Image


root_folder = "dataset/rsna-pneumonia-detection-challenge"

train_img_folder = Path(root_folder) / "stage_2_train_images"
train_resized_folder = Path(root_folder) / "train_images_512"

test_img_folder = Path(root_folder) / "test_images"
test_img_resized_folder = Path(root_folder) / "test_images_512"


def resize_and_save(inpath, outpath):

    for img_path in inpath.iterdir():
        dataset = pydicom.dcmread(img_path)
        image = dataset.pixel_array

        new_img_path = Path(outpath) / (img_path.stem + ".jpg")

        pil_img = Image.fromarray(image)
        pil_img.save(new_img_path)


if __name__ == "__main__":

    resize_and_save(train_img_folder, train_resized_folder)
    resize_and_save(test_img_folder, test_img_resized_folder)
