FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    MAVProxy==1.8.74 \
    pymavlink==2.4.49

CMD ["mavproxy.py", \
     "--master=tcp:127.0.0.1:5750", \
     "--out=udp:127.0.0.1:14552", \
     "--non-interactive"]
