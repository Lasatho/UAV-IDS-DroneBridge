FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tcpdump \
    curl \
    lsb-release \
    gnupg \
    && curl https://packages.osrfoundation.org/gazebo.gpg \
       --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
       http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y \
       gz-tools2 \
       gz-transport13-cli \
    && rm -rf /var/lib/apt/lists/*

# gz-transport Python bindings: the orchestrator publishes the time-varying
# wind in-process (the gz CLI takes ~2 s per call)
RUN apt-get update && apt-get install -y \
       python3-gz-transport13 \
       python3-gz-msgs10 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/orchestrator/requirements.txt
RUN pip3 install -r /opt/orchestrator/requirements.txt

COPY data/ /opt/orchestrator/

WORKDIR /opt/orchestrator

CMD ["python3", "orchestrator.py", \
     "--manifest", "/data/missions_manifest.csv", \
     "--output", "/data"]