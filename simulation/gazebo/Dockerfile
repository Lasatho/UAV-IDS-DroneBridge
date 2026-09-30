FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV GZ_SIM_SYSTEM_PLUGIN_PATH=/usr/local/lib/ardupilot_gazebo
ENV GZ_SIM_RESOURCE_PATH=/opt/ardupilot_gazebo/models:/opt/ardupilot_gazebo/worlds

# Gazebo Harmonic
RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    gnupg \
    && curl https://packages.osrfoundation.org/gazebo.gpg \
       --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
       http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gazebo-stable.list \
    && apt-get update && apt-get install -y gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# ardupilot_gazebo plugin
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libgz-sim8-dev \
    rapidjson-dev \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && git clone https://github.com/ArduPilot/ardupilot_gazebo /opt/ardupilot_gazebo_src \
    && cmake -S /opt/ardupilot_gazebo_src -B /opt/ardupilot_gazebo_src/build \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build /opt/ardupilot_gazebo_src/build --target install \
    && mkdir -p /opt/ardupilot_gazebo \
    && cp -r /opt/ardupilot_gazebo_src/models /opt/ardupilot_gazebo/models \
    && cp -r /opt/ardupilot_gazebo_src/worlds /opt/ardupilot_gazebo/worlds \
    && rm -rf /var/lib/apt/lists/*

# Wind: enable aerodynamic wind forcing on the iris fuselage and ship the
# custom world carrying the WindEffects plugin. The base_link needs the
# <enable_wind> flag for WindEffects to apply force to it.
RUN sed -i "/<link name='base_link'>/a\\      <enable_wind>true</enable_wind>" \
    /opt/ardupilot_gazebo/models/iris_with_standoffs/model.sdf
COPY worlds/iris_runway_wind.sdf /opt/ardupilot_gazebo/worlds/iris_runway_wind.sdf

CMD ["gz", "sim", "-s", "-r", "/opt/ardupilot_gazebo/worlds/iris_runway_wind.sdf"]
