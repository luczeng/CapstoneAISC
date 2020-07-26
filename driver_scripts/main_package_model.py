from Capstone.packaging.mlflow_packaging import package_model
from Capstone.configs.read_config import parse_config
import argparse
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Configs
    args = parse_args()
    cfg = parse_config(args.config_path)

    model_path = Path(cfg.save_path) / "ckp.pth"
    package_model(cfg.mlflow_pyfunc_model_path, str(model_path), "environment.yml")
