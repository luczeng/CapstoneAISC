import yaml


class parse_config:
    def __init__(self, config_path):
        """
            This class is just used as a container to avoid using dictionaries (less readable)
        """

        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        # General
        self.config_path = config_path

        # Network parameters
        self.net_type = config_dict["NET"]["NET_TYPE"]

        # Train parameters
        self.train_dataset_path = config_dict["TRAIN"]["TRAIN_DATASET_PATH"]
        self.train_label_path = config_dict["TRAIN"]["TRAIN_LABEL_PATH"]
        self.save_path = config_dict["TRAIN"]["SAVE_PATH"]
        self.n_epoch = config_dict["TRAIN"]["N_EPOCH"]
        self.lr = config_dict["TRAIN"]["LR"]
        self.loss_period = config_dict["TRAIN"]["LOSS_PERIOD"]
        self.mini_batch_size = config_dict["TRAIN"]["MINI_BATCH_SIZE"]
        self.saving_epoch = config_dict["TRAIN"]["SAVING_EPOCH"]
        self.load_checkpoint = bool(config_dict["TRAIN"]["LOAD_CHECKPOINT"])

        # Validation parameters
        self.use_validation = bool(config_dict["VALIDATION"]["USE_VALIDATION"])
        self.validation_period = config_dict["VALIDATION"]["VALIDATION_PERIOD"]

        # Test parameters
        self.test_dataset_path = config_dict["TEST"]["TEST_DATASET_PATH"]
        self.test_label_path = config_dict["TEST"]["TEST_LABEL_PATH"]
        self.weight_path = config_dict["TEST"]["WEIGHT_PATH"]

        # Mlflow
        self.log_weights = bool(config_dict["MLFLOW"]["LOG_WEIGHTS"])
