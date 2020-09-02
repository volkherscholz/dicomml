# FROM tensorflow/tensorflow:2.0.0-gpu-py3 AS base
FROM tensorflow/tensorflow:2.0.0-py3 AS base

# install gdcm
COPY . /dicomml
WORKDIR /dicomml

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install .

CMD ["run-trainer.sh"]
