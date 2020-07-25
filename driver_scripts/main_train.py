from Capstone.modeling.resnet import initalize_resnet
from Capstone.configs.read_config import parse_config
from Capstone.engine.train import train_loop
from Capstone.utils.nn_utils import print_training_info
from Capstone.utils.nn_utils import log_mlflow_param
from pathlib import Path
import argparse
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config_path", type=str, required=True)
    args = parser.parse_args()

    return args


def run_train(args):
    """
        Entry point for the training
    """

    # Configs
    cfg = parse_config(args.config_path)

    # Path
    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    ckp_path = Path(cfg.save_path) / "ckp.pth"
    save_path = Path(cfg.save_path) / "final_model.pth"

    # Net
    net = initalize_resnet(2, False, True)

    # Determine type(GPU or not)
    if torch.cuda.is_available():
        net.to(device=torch.device("cuda"))
        net_type = torch.cuda.FloatTensor
    else:
        net_type = torch.FloatTensor
    net = net.type(net_type)  # todo: find another way

    # Initlialization
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    criterion = CrossEntropyLoss()

    # Print net info and log parameters
    rsna_size = [1024, 1024]  # TODO: make this agnostic of dataset
    print_training_info(net, rsna_size)
    log_mlflow_param(cfg)

    # Training loop
    train_loop(cfg, ckp_path, save_path, net, net_type, optimizer, criterion)


if __name__ == "__main__":

    args = parse_args()

    run_train(args)
