FROM volkherscholz/tensorflow-gdcm

# install gdcm
COPY . /dicomml
WORKDIR /dicomml

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install .

CMD ["run-trainer.sh"]
