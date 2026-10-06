"""Exercise the ROS callback with real message types and deterministic adapters."""
import json
from types import SimpleNamespace
import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField

from ugv_self_supervised_traversability.pose_fusion import PoseEstimate
from ugv_self_supervised_traversability.terrain_analysis import TerrainConfig
from ugv_self_supervised_traversability.terrain_node import TerrainAnalyzer


def callback_result(imu_stamp):
    x,y=np.meshgrid(np.arange(-5.9,6,.08),np.arange(-5.9,6,.08))
    points=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-.4)]).astype('<f4')
    msg=PointCloud2();msg.header.frame_id='base_link';msg.header.stamp.sec=10
    msg.width=len(points);msg.height=1;msg.point_step=12;msg.row_step=len(points)*12;msg.data=points.tobytes()
    msg.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(('x','y','z'))]
    pose=PoseEstimate(10_000_000_000,0.,0.,0.,0.,0.,-np.radians(14),0.,'odometry+imu','odometry',imu_stamp)
    identity=TransformStamped();identity.transform.rotation.w=1.
    outputs=[];errors=[]
    adapter=SimpleNamespace(
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=10_000_000_000)),
        get_parameter=lambda name:SimpleNamespace(value={'use_imu_cloud_leveling':True,'odometry_z_valid':False}[name]),
        last_cloud_ns=None,period=.5,timeout=.25,base='base_link',world='odom',config=TerrainConfig(),
        poses=[(10_000_000_000,pose,dict(state='ASCENDING',source='imu_attitude_and_odometry_direction'),dict(state='LEVEL',source='odometry_z'))],
        fusion=SimpleNamespace(imu_status='valid'),
        tf=SimpleNamespace(lookup_transform=lambda *a,**k:identity),
        status_pub=SimpleNamespace(publish=lambda msg:outputs.append(json.loads(msg.data))),
        unknown=errors.append,publish_markers=lambda *args:None)
    TerrainAnalyzer.on_cloud(adapter,msg)
    assert not errors,errors
    return outputs[-1]


def test_imu_levels_cloud_without_3d_odometry_tf():
    status=callback_result(10_000_000_000)
    assert status['cloud_orientation_source']=='imu_roll_pitch+odometry_yaw_translation'
    assert status['forward']['status']=='GEOMETRY_PASS'
    assert status['reverse']['status']=='GEOMETRY_BLOCKED'
    assert status['imu_fresh'] is True
    assert status['forward']['max_uphill_deg']>13.5


def test_imu_must_be_fresh_relative_to_cloud_not_only_odometry():
    status=callback_result(9_600_000_000)
    assert status['cloud_orientation_source']=='recorded_world_tf'
    assert status['imu_fresh'] is False
    assert status['motion']['state']=='UNKNOWN'
    assert status['motion']['reason']=='imu_out_of_sync'


def test_bag_time_jump_clears_tf_and_sensor_state():
    cleared=[]
    adapter=SimpleNamespace(tf=SimpleNamespace(clear=lambda:cleared.append('tf')),
        poses=[1],motion=SimpleNamespace(history=[1]),
        fusion=SimpleNamespace(clear_sensor_cache=lambda:cleared.append('sensors')),
        last_cloud_ns=123,last_receipt=456,stale_sent=True)
    TerrainAnalyzer.on_time_jump(adapter,None)
    assert cleared==['tf','sensors']
    assert adapter.poses==[] and adapter.motion.history==[]
    assert adapter.last_cloud_ns is None and adapter.last_receipt is None
    assert adapter.stale_sent is False
