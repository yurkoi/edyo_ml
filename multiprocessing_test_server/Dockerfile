FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get -y --no-install-recommends install libtool && \
    apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
  apt-get -y --no-install-recommends install \    
  python3.6 python3.6-dev python-dev python3-dev \
  python3-pip \
        python3-gi \
        python3-gst-1.0 \
        libcairo2-dev \ 
        libgstreamer1.0-0 \
        libgirepository1.0-dev
 
COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "websocket_server.py"]
