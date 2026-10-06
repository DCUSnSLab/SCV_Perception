from types import SimpleNamespace
import numpy as np
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_ros import LookupException

from ugv_self_supervised_traversability.pose_fusion import PoseFusion, pointcloud_to_xyz
from ugv_self_supervised_traversability.utils.transforms import quaternion_from_euler


def fusion(buffer):
    return PoseFusion(buffer,'odom',base_frame='base_link',use_imu_orientation=True,
        imu_use_yaw=False,imu_timeout_seconds=.2,use_lidar_ground_height=False,
        pointcloud_timeout_seconds=.2,lidar_ground_radius=1.2,
        lidar_ground_min_points=20,lidar_ground_percentile=20)


def imu(roll=0,pitch=0):
    msg=Imu();msg.header.frame_id='imu_link';msg.header.stamp.sec=10
    q=quaternion_from_euler(roll,pitch,0)
    msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w=map(float,q)
    return msg


def odom(second=10):
    msg=Odometry();msg.header.stamp.sec=second;msg.pose.pose.orientation.w=1.
    return msg


def test_imu_mounting_rotation_and_staleness():
    mount=TransformStamped()
    q=quaternion_from_euler(.3,0,0)
    mount.transform.rotation.x,mount.transform.rotation.y,mount.transform.rotation.z,mount.transform.rotation.w=map(float,q)
    f=fusion(SimpleNamespace(lookup_transform=lambda *a,**k:mount))
    f.update_imu(imu(roll=.3))
    pose=f.fuse_odometry(odom())
    assert abs(pose.roll)<1e-8 and abs(pose.pitch)<1e-8
    assert pose.orientation_source=='odometry+imu'
    assert f.fuse_odometry(odom(11)).orientation_source=='odometry'


def test_invalid_imu_and_missing_mount_do_not_apply_orientation():
    def missing(*a,**k):raise LookupException('missing')
    f=fusion(SimpleNamespace(lookup_transform=missing))
    f.update_imu(imu(.5))
    assert f.imu_status=='mounting_tf_missing'
    assert f.fuse_odometry(odom()).orientation_source=='odometry'
    m=imu();m.orientation_covariance[0]=-1.
    f.update_imu(m)
    assert f.imu_status=='invalid'
    m=imu();m.orientation.x=float('nan');f.update_imu(m)
    assert f.fuse_odometry(odom()).orientation_source=='odometry'


def test_pointcloud_organized_padding_and_nonfinite():
    msg=PointCloud2();msg.width=2;msg.height=2;msg.point_step=16;msg.row_step=40
    msg.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(('x','y','z'))]
    raw=bytearray(80)
    for row in range(2):
        for col in range(2):
            p=np.array([row,col,1.],dtype='<f4')
            raw[row*40+col*16:row*40+col*16+12]=p.tobytes()
    msg.data=bytes(raw)
    assert np.array_equal(pointcloud_to_xyz(msg),[[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
