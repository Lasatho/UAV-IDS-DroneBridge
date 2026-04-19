FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libpcap-dev \
    libusb-1.0-0-dev \
    zlib1g-dev \
    iw \
    wireless-tools \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/Lasatho/DroneBridge.git /opt/dronebridge_src \
    && cmake -S /opt/dronebridge_src -B /opt/dronebridge_src/build \
    && cmake --build /opt/dronebridge_src/build \
    && mkdir -p /opt/dronebridge \
    && cp /opt/dronebridge_src/build/proxy/db_proxy /opt/dronebridge/db_proxy \
    && rm -rf /opt/dronebridge_src

CMD ["/opt/dronebridge/db_proxy", "-n", "wlan1", "-m", "m", "-c", "200", "-p", "5750", "-d", "1"]
