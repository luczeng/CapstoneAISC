FROM nvidia/cuda:10.2-base
CMD nvidia-smi

# Set the working dir to root
WORKDIR /root

# System updates and configurations
RUN apt-get update && apt-get -y --no-install-recommends install \
    ca-certificates \
    git \
    ssh \
    wget && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to include Miniconda
ENV PATH /root/miniconda/bin:$PATH

# Copy over model and api code into the image
COPY ./api /root
COPY ./flask_api /root
COPY setup.py /root

# Create a conda environment from the specified conda.yml
RUN conda env create --file /root/conda.yaml

# Add to bashrc
RUN echo "source activate capstone" >> .bashrc

# Install packages and pip libraries into the conda environment
RUN /bin/bash -c "source activate capstone" 
RUN pip install --upgrade pip setuptools
RUN pip install -e .

RUN ["chmod","+x", "/root/start.sh"]

# Start the api
#ENTRYPOINT ["/root/start.sh"]
