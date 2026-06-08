# One tracked 3D object

# Tracking information
uint32 track_id
uint32 age                 # number of frames since this track was created
uint32 missed_count        # number of consecutive missed frames
bool is_confirmed          # true if tracker considers this track reliable

uint16 class_id
string class_name
float32 confidence         # detection or tracking confidence, 0.0 ~ 1.0

# 3D bounding box
# pose.position: center of 3D bounding box
# pose.orientation: orientation of 3D bounding box
geometry_msgs/PoseWithCovariance bbox_pose

# size.x: length, size.y: width, size.z: height [m]
geometry_msgs/Vector3 bbox_size

# Object velocity
# linear.x, linear.y, linear.z [m/s]
# angular.x, angular.y, angular.z [rad/s]
geometry_msgs/TwistWithCovariance twist

# Optional state
float32 existence_probability   # probability that object really exists, 0.0 ~ 1.0
bool is_dynamic                 # true if object is moving