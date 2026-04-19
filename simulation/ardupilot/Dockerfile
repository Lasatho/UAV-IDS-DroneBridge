FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ArduPilot
RUN git clone https://github.com/ArduPilot/ardupilot /opt/ardupilot \
    && cd /opt/ardupilot \
    && git submodule update --init --recursive \
    && Tools/environment_install/install-prereqs-ubuntu.sh -y

RUN cd /opt/ardupilot \
    && ./waf configure --board sitl \
    && ./waf copter

CMD ["/opt/ardupilot/build/sitl/bin/arducopter", \
     "--model", "JSON", \
     "--speedup", "1", \
     "--defaults", "/opt/ardupilot/Tools/autotest/default_params/copter.parm,/opt/ardupilot/Tools/autotest/default_params/gazebo-iris.parm", \
     "--sim-address=127.0.0.1", \
     "-I0"]
