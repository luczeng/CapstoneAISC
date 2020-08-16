# CapstoneAISC
 This project allows to train, package and deploy a model for pneumonia classification. The models are trained on the
 RSNA dataset.  

 Note: this project was not unit tested. Even if it was shown to work, we do not guarantee its usage.

# Installation
Create a conda environment and type `pip install -e .`

# Data preparation
We offer some helper script to prepare the dataset:
- [Train-test split](driver_scripts/main_split_train_test.py)
- [Resize images](driver_scripts/main_resize_imgs.py)
- [Downsample the negatives](driver_scripts/main_create_balanced_dataset.py)
- [Count the number of positives and negatives](driver_scripts/main_analyse_dataset.py)

# Training
The entry point to the training script is [here](driver_scripts/main_train.py). You can specify the parameters in your own config file. We provide an example [here](Capstone/configs/config_RSNA.yml).

# Packaging, containerizing and deploy
From your conda environment, follow the following steps.  

1. To package the model in a self contained folder:  
`python driver_scripts/main_package_model.py -c Capstone/configs/config_RSNA.yml`
2. To create a docker container:  
`docker build -t flask_api`
3. To serve the container:  
`docker run -it --rm --gpus all --name flask_cont -p 5000:5000 flask_api`
