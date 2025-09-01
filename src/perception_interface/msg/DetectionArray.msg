# Array of detection results
std_msgs/Header header

# Source image info
sensor_msgs/Image source_image
string model_name
string model_version

# Detection results
DetectionResult[] detections