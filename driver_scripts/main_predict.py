from Capstone.configs.read_config import parse_config
from Capstone.io.io import load_folder
import argparse
import pandas as pd
import mlflow.pyfunc
import pydicom


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("-i", "--input_image", type=str, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Configs
    args = parse_args()
    cfg = parse_config(args.config_path)

    # Load model
    loaded_model = mlflow.pyfunc.load_model(cfg.mlflow_pyfunc_model_path)

    # Load image
    imgs = load_folder(args.input_image)

    # Create panda dataframe
    predictions = loaded_model.predict(pd.DataFrame(imgs))

    print(type(predictions), predictions)
