
FROM python:3.7.7-stretch AS BASE

RUN apt-get update \
    && apt-get --assume-yes --no-install-recommends install \
        build-essential \
        curl \
        git \
        jq \
        libgomp1 \
        vim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install rasa==3.1.0
RUN pip install transformers==4.20.1
RUN pip install torch==1.12.0
RUN pip install datasets==2.3.2
RUN pip install sentencepiece==0.1.96
RUN pip install sentence-transformers==2.2.2
RUN pip install sanic-plugin-toolkit==1.2.0


#Optional step

ADD config.yml config.yml
ADD domain.yml domain.yml
ADD credentials.yml credentials.yml
ADD endpoints.yml endpoints.yml