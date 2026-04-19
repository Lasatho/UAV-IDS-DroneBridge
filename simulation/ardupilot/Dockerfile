FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    sudo \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ArduPilot install script refuses to run as root
RUN useradd -m ardupilot && echo "ardupilot ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ardupilot
ENV USER=ardupilot
ENV TZ=Europe/Vienna
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ | sudo tee /etc/timezone
WORKDIR /home/ardupilot

RUN git clone https://github.com/ArduPilot/ardupilot /home/ardupilot/ardupilot \
    && cd /home/ardupilot/ardupilot \
    && git submodule update --init --recursive

RUN cd /home/ardupilot/ardupilot \
    && Tools/environment_install/install-prereqs-ubuntu.sh -y

ENV PATH="/home/ardupilot/ardupilot/Tools/autotest:${PATH}"
ENV PATH="/home/ardupilot/.local/bin:${PATH}"

RUN cd /home/ardupilot/ardupilot \
    && ./waf configure --board sitl \
    && ./waf copter

CMD ["/home/ardupilot/ardupilot/build/sitl/bin/arducopter", \
     "--model", "JSON", \
     "--speedup", "1", \
     "--defaults", "/home/ardupilot/ardupilot/Tools/autotest/default_params/copter.parm,/home/ardupilot/ardupilot/Tools/autotest/default_params/gazebo-iris.parm", \
     "--sim-address=127.0.0.1", \
     "-I0"]
