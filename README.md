# CapstoneAISC
 This project allows to train, package and deploy a model for pneumonia classification. The models are trained on the
 RSNA dataset.

# Installation
Create a conda environment and type `pip install -e .`

# Training
TBD

# Packaging, containerizing and deploy
From your conda environment, follow the following steps.  

1. To package the model in a self contained folder:  
`python driver_scripts/main_package_model.py -c Capstone/configs/config_RSNA.yml`
2. To create a docker container:  
`docker build -t flask_api`
3. To serve the container:  
`docker run -it --rm --gpus all --name flask_cont -p 5000:5000 flask_api`
