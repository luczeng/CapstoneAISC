from Capstone.data.prepare_dataset import split_test_train, save_labels_to_disk
from pathlib import Path

# default argument
root_folder = "dataset/rsna-pneumonia-detection-challenge"
n_test_imgs = 500

train_img_folder = Path(root_folder) / "stage_2_train_images"
label_file = Path(root_folder) / "stage_2_train_labels.csv"
cleaned_train_lbl_path = Path(root_folder) / "cleaned_train_labels.csv"

test_img_folder = Path(root_folder) / "test_images"
test_lbl_path = Path(root_folder) / "test_labels.csv"

if __name__ == "__main__":
    """
        Helper script to extract some images from the train folder and move them to the test folder
    """
    train_labels, test_labels = split_test_train(train_img_folder, label_file, test_img_folder, n_test_imgs)

    # Write test labels to disk
    save_labels_to_disk(test_labels, test_lbl_path)

    # Write train labels to disk
    save_labels_to_disk(train_labels, cleaned_train_lbl_path)
