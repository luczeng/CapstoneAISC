from torch.utils.data import DataLoader
from Capstone.utils.nn_utils import load_checkpoint, save_checkpoint, define_checkpoint
from Capstone.data.datasets import DatasetRSNA
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
    if ckp_path.exists() and cfg.load_checkpoint:
        start = load_checkpoint(ckp_path, net, optimizer)

    # Data
    dataset = DatasetRSNA(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.mini_batch_size, shuffle=True)

    # Training loop
    iterations = 0
    running_loss = 0.0
    for epoch in range(start, cfg.n_epoch):
        for idx, batch in enumerate(dataloader):

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
            if (epoch % cfg.loss_period == (cfg.loss_period - 1)) & (epoch != 0):

                # Mlflow loggin
                mlflow.log_metric("train_loss", running_loss / iterations, step=epoch + idx)

                # Print training info
                running_loss, iterations = print_info_small_dataset(
                    running_loss, iterations, epoch, idx, len(dataset), cfg
                )

            # Run evaluation
            if (epoch % cfg.validation_period == cfg.validation_period - 1) & epoch != 0:
                pass

            if epoch % cfg.saving_epoch == cfg.saving_epoch - 1:
                # Checkpoint, save checkpoint to disck
                ckp = define_checkpoint(net, optimizer, epoch)
                save_checkpoint(ckp, ckp_path)
