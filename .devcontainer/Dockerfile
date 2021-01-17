FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN conda install -c conda-forge gdcm && \
    apt-get update && apt-get install -y git ssh-client
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install jupyter matplotlib plotly flake8
