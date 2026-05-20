FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/orchestrator/requirements.txt
RUN pip3 install -r /opt/orchestrator/requirements.txt

COPY data/ /opt/orchestrator/

WORKDIR /opt/orchestrator

CMD ["python3", "orchestrator.py", \
     "--manifest", "/data/missions_manifest.csv", \
     "--output", "/data"]