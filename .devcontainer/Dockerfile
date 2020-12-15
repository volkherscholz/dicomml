FROM volkherscholz/tensorflow-gdcm

RUN apt-get update && apt-get install -y git ssh-client
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && \
    pip install jupyter matplotlib plotly flake8
