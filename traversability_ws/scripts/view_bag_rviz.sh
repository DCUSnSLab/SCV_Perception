#!/usr/bin/env bash
# Run only this review's processes; closing RViz stops its bag and analyzer.
set -e
review_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
review_bag="${1:-$review_ws/rosbag2_2026_08_13-13_55_28}"
review_rate="${REVIEW_RATE:-0.5}"
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${REVIEW_ROS_DOMAIN_ID:-170}"
export ROS_LOCALHOST_ONLY=1
export PYTHONPATH="$review_ws/src/ugv_self_supervised_traversability:${PYTHONPATH:-}"
export ROS_LOG_DIR="$(mktemp -d /tmp/traversability-rviz-XXXXXX)"
review_config="$review_ws/src/ugv_self_supervised_traversability/config"
if [ "${1:-}" = "--check" ]; then
    exec /usr/bin/python3 "$review_ws/scripts/check_rviz_review.py"
fi
if [ "${1:-}" = "--selftest" ]; then
    exec /usr/bin/python3 "$review_ws/scripts/smoke_terrain_ros.py"
fi
review_node_pid=""
review_bag_pid=""
review_rviz_pid=""
cleanup() {
    for review_pid in "$review_rviz_pid" "$review_bag_pid" "$review_node_pid"; do
        if [ -n "$review_pid" ] && kill -0 "$review_pid" 2>/dev/null; then
            kill -INT "$review_pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT TERM
/usr/bin/python3 -c 'from ugv_self_supervised_traversability.terrain_node import main; main()' \
    --ros-args --params-file "$review_config/traversability.yaml" -p use_sim_time:=true \
    >"$ROS_LOG_DIR/terrain.log" 2>&1 &
review_node_pid=$!
rviz2 -d "$review_config/terrain_review.rviz" --ros-args -p use_sim_time:=true \
    >"$ROS_LOG_DIR/rviz.log" 2>&1 &
review_rviz_pid=$!
ros2 bag play "$review_bag" --clock 30 --loop --rate "$review_rate" --delay 3 \
    --read-ahead-queue-size 100 --disable-keyboard-controls \
    --topics /odom /tf /tf_static /velodyne_points /traversability/trajectory /traversability/footprint \
    >"$ROS_LOG_DIR/bag.log" 2>&1 &
review_bag_pid=$!
printf 'RViz review running: domain=%s rate=%s logs=%s\n' "$ROS_DOMAIN_ID" "$review_rate" "$ROS_LOG_DIR"
printf 'Processes: rviz=%s bag=%s terrain=%s\n' "$review_rviz_pid" "$review_bag_pid" "$review_node_pid"
wait "$review_rviz_pid"
