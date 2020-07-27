from Capstone.configs.read_config import parse_config
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

if __name__ == '__main__':

    # Configs
    args = parse_args()
    cfg = parse_config(args.config_path)

    # Load model
    loaded_model = mlflow.pyfunc.load_model(cfg.mlflow_pyfunc_model_path)

    # Load image
    data = dataset = pydicom.dcmread(args.input_image)
    img = data.pixel_array

    # Create panda dataframe
    predictions = loaded_model.predict(pd.DataFrame(img))

    print(type(predictions), predictions)
