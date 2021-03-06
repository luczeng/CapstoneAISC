import argparse
from pathlib import Path
from Capstone.modeling.resnet import initalize_resnet
from Capstone.utils.nn_utils import load_checkpoint
from Capstone.utils.evaluation_utils import evaluate_model
import torch
from PIL import Image
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--ckp_path", type=str, required=True)
    parser.add_argument("-l", "--label_file_path", type=str, required=True)
    parser.add_argument("-i", "--image_folder", type=str, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    # Configs
    args = parse_args()

    # Net
    net = initalize_resnet(2, False, True)

    load_checkpoint(args.ckp_path, net)

    # Determine type(GPU or not)
    if torch.cuda.is_available():
        net.to(device=torch.device("cuda"))
        net_type = torch.cuda.FloatTensor
    else:
        net_type = torch.FloatTensor
    net = net.type(net_type)  # todo: find another way
    net.eval()

    load_checkpoint(args.ckp_path, net)
    net.eval()

    error = evaluate_model(net, args.image_folder, args.label_file_path, 1, net_type)
    print("The error is: ", error)
