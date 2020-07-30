from Capstone.utils.nn_utils import load_checkpoint, save_checkpoint, define_checkpoint
from Capstone.data.datasets import DatasetRSNA
from Capstone.utils.nn_utils import print_info
from Capstone.utils.evaluation_utils import evaluate_model
from torch.utils.data import DataLoader
import torch
import mlflow
import mlflow.pytorch


def train_loop(cfg, ckp_path, save_path, net, net_type, optimizer, criterion):
    """
        :param cfg
        :param ckp_path path to checkpoint
        :param save_path path to final model
        :param net
        :param net_type cpu or gpu
    """

    # Resume
    start = 0
    # if ckp_path.exists() and cfg.load_checkpoint:
        # start = load_checkpoint(ckp_path, net, optimizer)

    # Data
    dataset = DatasetRSNA(net_type, cfg.train_dataset_path, cfg.train_label_path)
    dataloader = DataLoader(dataset, batch_size=cfg.mini_batch_size, shuffle=True)

    # Training loop
    iterations = 0
    running_loss = 0.0
    epoch = 0
    for epoch in range(start, cfg.n_epoch):
        idx = 0
        for batch in dataloader:

            # GPU
            net.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            x = net.forward(batch["image"])

            # Backward pass
            loss = criterion(x, batch["label"])
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            iterations += 1

            # Print info, logging
            idx += 1
            if (idx % cfg.loss_period == (cfg.loss_period - 1)) & (idx != 0):
                mlflow.log_metric(
                    "train_loss", running_loss / (cfg.loss_period * cfg.mini_batch_size), step=epoch + idx
                )
                # Print training info
                running_loss = print_info(running_loss, iterations, epoch, idx, len(dataset), cfg)
                print("\t", torch.argmax(x, axis=1), "\t", batch["label"])

            # Run evaluation
            if (idx % cfg.validation_period == (cfg.validation_period - 1)) & idx != 0:
                accuracy = evaluate_model(net, cfg.test_dataset_path, cfg.test_label_path, cfg, net_type)
                mlflow.log_metric("accuracy", accuracy.item(), step=epoch + idx)
                print(
                    "Epoch : %d/%d, iteration: %d/%5d || Accuracy: %.3f"
                    % (epoch, cfg.n_epoch, idx + 1, len(dataset) / cfg.mini_batch_size, accuracy)
                )

            if idx % cfg.saving_epoch == cfg.saving_epoch - 1:
                # Checkpoint, save checkpoint to disck
                ckp = define_checkpoint(net, optimizer, epoch)
                save_checkpoint(ckp, ckp_path)

        epoch += 1

    # Run evaluation on test set
