FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN conda install -c conda-forge gdcm && \
    pip install --no-cache-dir -r requirements.txt

COPY . /dicomml
WORKDIR /dicomml

RUN pip install .

ENTRYPOINT ["python"]
