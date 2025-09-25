# Multiple waypoints message for smooth path following
# Provides current goal and lookahead waypoints to prevent speed reduction near goals

std_msgs/Header header

# Current target waypoint (what we're heading to now)
geometry_msgs/PoseStamped current_goal
bool current_goal_reverse_heading        # True if current goal requires reverse heading
int32 current_goal_node_type            # Node type of current goal

# Next waypoints for lookahead (typically 1-3 waypoints ahead)
geometry_msgs/PoseStamped[] next_waypoints
bool[] next_waypoints_reverse_heading    # True if corresponding waypoint requires reverse heading
int32[] next_waypoints_node_types        # Node types of next waypoints

# Path information
string path_id                    # Path identifier
int32 current_waypoint_index      # Index of current goal in full path
int32 total_waypoints             # Total number of waypoints in path
bool is_final_waypoint            # True if current_goal is the final destination