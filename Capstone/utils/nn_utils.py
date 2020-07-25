import torch
import mlflow
from pathlib import Path
from torchsummary import summary


def print_training_info(net, img_size):
    print("\nNetwork information: \n")
    print(net, "\n")
    summary(net, (1, img_size[0], img_size[1]))
    print("\n")


def print_info(running_loss, iterations, epoch, idx, dataset_len, config):
    """
        Convenience function to print info each specified period. Resets the running loss.
        For small dataset scripts.
        :param running_loss
        :param iterations number of iterations to divide the loss with
        :param epoch
        :param idx iteration index (for one epoch)
        :param dataset_len number of points in the dataset
        :param config
        :return running_loss, iterations resetted to 0
        TODO: Move to other file (training_utils.py?)
    """

    # Prints loss
    print(
        "Epoch : %d/%d, iteration: %d/%5d || loss: %.3f"
        % (epoch, config.n_epoch, idx + 1, dataset_len / config.mini_batch_size, running_loss / config.loss_period,)
    )

    return 0.0


def log_mlflow_param(config):
    """
        Log mlflow metrics.
        Logs the whole config file and some specific parameters that are to be seen at first by the user
        :param config parse_config file
    """

    mlflow.log_artifact(config.config_path)
    mlflow.log_param("lr", config.lr)
    mlflow.log_param("dataset_name", Path(config.train_dataset_path).name)


def save_checkpoint(state, ckp_path):
    torch.save(state, ckp_path)


def load_checkpoint(ckp_path, model, optimizer):
    """
        Updates the model and optimizers to the checkpoint's state
        :param ckp_path path to checkpoint (.pth file)
        :param model
        :param optimizer
        :return: checkpoint's epoch
    """
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


def define_checkpoint(model, optimizer, epoch):
    """
    TODO
    """

    return {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}


def load_model(model_path: str, model):
    model.load_state_dict(checkpoint["state_dict"])
