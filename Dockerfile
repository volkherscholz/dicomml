# FROM tensorflow/tensorflow:2.0.0-gpu-py3 AS base
FROM tensorflow/tensorflow:2.0.0-py3 AS base

# install gdcm
COPY . /tmpinstall
RUN apt-get update \ 
    && apt-get install -y libgdcm2.8 python-gdcm \
    && dpkg -i /tmpinstall/build_1-1_amd64.deb \
    && apt-get install -f \
    && cp /usr/local/lib/gdcm.py /usr/local/lib/python3.6/dist-packages/. \
    && cp /usr/local/lib/gdcmswig.py /usr/local/lib/python3.6/dist-packages/. \
    && cp /usr/local/lib/_gdcmswig.so /usr/local/lib/python3.6/dist-packages/. \
    && cp /usr/local/lib/libgdcm* /usr/local/lib/python3.6/dist-packages/. \
    && ldconfig

# install package & requirements
RUN pip install -r /tmpinstall/requirements.txt \
    && pip install /tmpinstall \
    && rm -r /tmpinstall \
    && mkdir -p /var/secrets/google

# set environment variable for key
ENV GOOGLE_APPLICATION_CREDENTIALS /var/secrets/google/key.json

ENTRYPOINT ["python3", "-m", "dicomml.tasks.training"]

#################################################################################

FROM base AS develop

RUN apt-get install -y wget curl nodejs npm \
    && pip install jupyterlab matplotlib

# create dicomml user and give him sudo rights
RUN useradd -m -s /bin/bash -N -u 1000 -g 100 dicomml \
    && apt-get install -y sudo \
    && echo "dicomml ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user

USER dicomml
WORKDIR /home/dicomml
RUN mkdir ./work
RUN mkdir ./examples

ENV SHELL /bin/bash
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser"]
