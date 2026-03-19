std_msgs/Header header
uint32 id             # 트래킹 ID
string label          # 클래스 (Car, Pedestrian)
float32 score         # 신뢰도
float32[3] pose       # x, y, z (중심)
float32[3] dimensions # l, w, h (크기)
float32 yaw           # 회전각