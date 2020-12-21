FROM volkherscholz/tensorflow-gdcm:tf-2.4.0-gpu

# install gdcm
COPY . /dicomml
WORKDIR /dicomml

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install .

CMD ["run-trainer.sh"]
