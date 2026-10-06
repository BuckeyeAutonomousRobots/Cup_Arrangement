ARG ROS_BASE_IMAGE=ros:humble-ros-base
FROM ${ROS_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git \
    sudo \
    build-essential \
    xfce4 \
    xfce4-terminal \
    dbus-x11 \
    x11vnc \
    xvfb \
    novnc \
    websockify \
    xterm \
    net-tools \
    procps \
    python3-colcon-common-extensions \
    python3-pip \
    python3-yaml \
    python3-numpy \
    python3-scipy \
    python3-pytest \
    ros-humble-ur-description \
    ros-humble-xacro \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-robot-state-publisher \
    ros-humble-controller-manager \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ur-msgs \
    ros-humble-ur-robot-driver \
    ros-humble-moveit \
    ros-humble-moveit-servo \
    ros-humble-ros-gz \
    ros-humble-ros-gz-sim \
    ros-humble-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# Python deps used by teleop_bridge
RUN python3 -m pip install --no-cache-dir transforms3d

ARG USERNAME=robot
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY scripts/start_desktop.sh /start_desktop.sh
RUN chmod +x /start_desktop.sh
COPY scripts/gz /usr/local/bin/gz
RUN chmod +x /usr/local/bin/gz

RUN mkdir -p /workspace/src && chown -R ${UID}:${GID} /workspace

USER ${USERNAME}
WORKDIR /home/${USERNAME}
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc \
    && echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> ~/.bashrc

WORKDIR /workspace
RUN mkdir -p src

ENTRYPOINT ["/entrypoint.sh"]
