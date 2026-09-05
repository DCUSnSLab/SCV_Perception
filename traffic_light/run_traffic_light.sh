#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/ssc/SSC/src/perception/install/setup.bash

exec ros2 launch mando_tools traffic_light.launch.py "$@"
