FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt requirements.txt

RUN conda install -y -n base -c conda-forge gdcm && \
    pip install -r requirements.txt

COPY . /dicomml
WORKDIR /dicomml

RUN pip install .

ENTRYPOINT ["python"]
