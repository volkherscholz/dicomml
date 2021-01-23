FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt requirements.txt

RUN conda install -c conda-forge gdcm && \
    conda install --file requirements.txt

COPY . /dicomml
WORKDIR /dicomml

RUN pip install .

ENTRYPOINT ["python"]
