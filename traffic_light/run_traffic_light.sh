#!/usr/bin/env bash
set -eo pipefail

traffic_light_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ssc_root="$(cd -- "$traffic_light_root/../../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$ssc_root/install/setup.bash"

export MANDO_WS="$traffic_light_root"
export PYTHONPATH="$MANDO_WS/.deps${PYTHONPATH:+:$PYTHONPATH}"

python3 -c 'import torch; print(f"tl_fusion device check: cuda={torch.cuda.is_available()}" + (f", gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))'

exec ros2 launch mando_tools traffic_light.launch.py "$@"
