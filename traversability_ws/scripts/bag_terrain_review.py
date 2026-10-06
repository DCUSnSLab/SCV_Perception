"""Run current terrain geometry on sampled bag clouds and export a 3D review.

Requires the ROS Python environment, numpy and plotly. Reads SQLite read-only;
uses recorded sensor timestamps and TF, without publishing ROS topics.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import plotly.graph_objects as go
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer, TransformException

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/ugv_self_supervised_traversability'))
from ugv_self_supervised_traversability.pose_fusion import pointcloud_to_xyz, stamp_to_ns
from ugv_self_supervised_traversability.terrain_analysis import TerrainConfig, analyze_terrain
from ugv_self_supervised_traversability.utils.transforms import transform_matrix, transform_points, euler_from_quaternion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag',type=Path)
    parser.add_argument('--output',type=Path,default=Path('test_results/terrain_3d'))
    parser.add_argument('--interval',type=float,default=10.0)
    args=parser.parse_args()
    if args.interval<=0: parser.error('interval must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(f'file:{args.bag.resolve()}?mode=ro',uri=True)
    topics={name:(idx,get_message(typ)) for idx,name,typ in db.execute('SELECT id,name,type FROM topics')}
    start,end=db.execute('SELECT MIN(timestamp),MAX(timestamp) FROM messages').fetchone()
    tf=Buffer(cache_time=Duration(seconds=(end-start)/1e9+60))
    for name in ('/tf_static','/tf'):
        if name not in topics: continue
        idx,cls=topics[name]
        for (blob,) in db.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',(idx,)):
            for transform in deserialize_message(blob,cls).transforms:
                (tf.set_transform_static if name=='/tf_static' else tf.set_transform)(transform,'recorded_bag')
    idx,cls=topics['/odom']
    odom=[]
    for (blob,) in db.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',(idx,)):
        m=deserialize_message(blob,cls)
        if m.header.frame_id!='odom': raise ValueError('Review currently requires odom world frame')
        p,q=m.pose.pose.position,m.pose.pose.orientation
        roll,pitch,yaw=euler_from_quaternion([q.x,q.y,q.z,q.w])
        odom.append([stamp_to_ns(m.header.stamp),p.x,p.y,p.z,roll,pitch,yaw])
    odom=np.asarray(odom)
    idx,cls=topics['/velodyne_points']
    selected=[]; previous=-float('inf')
    for mid,stamp in db.execute('SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp',(idx,)):
        if (stamp-previous)/1e9>=args.interval:
            selected.append((mid,stamp)); previous=stamp
    config=TerrainConfig()
    frames=[]; records=[]; clouds=[]; surfaces=[]; colors=[]
    for mid,stamp in selected:
        msg=deserialize_message(db.execute('SELECT data FROM messages WHERE id=?',(mid,)).fetchone()[0],cls)
        sensor_stamp=stamp_to_ns(msg.header.stamp)
        pose=odom[np.argmin(np.abs(odom[:,0]-sensor_stamp))]
        relative=(stamp-start)/1e9
        record=dict(time_seconds=relative,sensor_timestamp_ns=sensor_stamp)
        if abs(pose[0]-sensor_stamp)>0.25e9:
            records.append(dict(record,status='UNKNOWN',reason='odometry_out_of_sync')); continue
        try:
            transform=tf.lookup_transform('odom',msg.header.frame_id,Time.from_msg(msg.header.stamp))
        except TransformException as error:
            records.append(dict(record,status='UNKNOWN',reason=str(error))); continue
        p,q=transform.transform.translation,transform.transform.rotation
        world=transform_points(pointcloud_to_xyz(msg),transform_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w]))
        before=time.perf_counter()
        result=analyze_terrain(world,pose[1],pose[2],pose[6],config)
        elapsed=time.perf_counter()-before
        cells=result.centers[result.observed].copy()
        x,y=cells[:,0].copy(),cells[:,1].copy()
        cells[:,0]=pose[1]+np.cos(pose[6])*x-np.sin(pose[6])*y
        cells[:,1]=pose[2]+np.sin(pose[6])*x+np.cos(pose[6])*y
        states=result.status[0][result.observed]
        palette=np.array(['#91a0b6','#31d99b','#ff5a68'])
        local=world[np.hypot(world[:,0]-pose[1],world[:,1]-pose[2])<=config.radius]
        local=local[::max(1,int(np.ceil(len(local)/5000)))]
        clouds.append(local.round(3).tolist());surfaces.append(cells.round(3).tolist());colors.append(palette[states].tolist())
        record.update(forward=result.corridor(),reverse=result.corridor(-1),
            analysis_seconds=elapsed,input_points=len(world),observed_cells=int(result.observed.sum()),
            forward_cell_counts={name:int(np.sum(result.status[0][result.observed]==s)) for s,name in enumerate(['UNKNOWN','GEOMETRY_PASS','GEOMETRY_BLOCKED'])})
        records.append(record)
        def traces():
            return [go.Scatter3d(x=local[:,0],y=local[:,1],z=local[:,2],mode='markers',
                    marker=dict(size=1.5,color=local[:,2],colorscale='Viridis',opacity=.35),name='LiDAR (world z)'),
                go.Scatter3d(x=cells[:,0],y=cells[:,1],z=cells[:,2],mode='markers',
                    marker=dict(size=4,color=palette[states]),name='Forward geometry',
                    text=[['UNKNOWN','GEOMETRY_PASS','GEOMETRY_BLOCKED'][s] for s in states]),
                go.Scatter3d(x=[pose[1]],y=[pose[2]],z=[pose[3]],mode='markers',marker=dict(size=7,color='#ffd37a'),name='UGV odometry')]
        caption=f'{relative:.1f}s | Forward: {record["forward"]["status"]} | Reverse: {record["reverse"]["status"]}'
        frames.append(go.Frame(name=str(len(frames)),data=traces(),layout=go.Layout(title=caption)))
        print(f'{relative:.1f}s: {len(local)} displayed points, {record["observed_cells"]} cells, {elapsed:.3f}s analysis',flush=True)
    db.close()
    imu_present='/imu/data' in topics
    report=dict(bag=str(args.bag),limits=asdict(config),sample_interval_seconds=args.interval,
        imu_recorded=imu_present,odom_z_range_m=[float(odom[:,3].min()),float(odom[:,3].max())],
        note='Current geometry code on recorded TF. IMU motion not evaluated. Flat odometry/incorrect gravity alignment can distort slopes.',scans=records)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if not frames: raise RuntimeError('No cloud could be transformed; see report.json')
    fig=go.Figure(data=frames[0].data,frames=frames)
    fig.update_layout(template='plotly_dark',title=frames[0].layout.title,
        scene=dict(xaxis_title='World X (m)',yaxis_title='World Y (m)',zaxis_title='World Z (m)',aspectmode='data'),
        margin=dict(l=0,r=0,b=120,t=80),height=850,
        updatemenus=[dict(type='buttons',buttons=[dict(label='Play',method='animate',args=[None,dict(frame=dict(duration=600,redraw=True),fromcurrent=True,transition=dict(duration=0))]),dict(label='Pause',method='animate',args=[[None],dict(mode='immediate',frame=dict(duration=0,redraw=False))])],x=0,y=0)],
        sliders=[dict(active=0,steps=[dict(label=f'{records[i]["time_seconds"]:.0f}s' if len(records)==len(frames) else str(i),method='animate',args=[[f.name],dict(mode='immediate',frame=dict(duration=0,redraw=True),transition=dict(duration=0))]) for i,f in enumerate(frames)],y=0)],
        annotations=[dict(text='Green: geometry pass / Red: blocked / Gray: unknown. Example limits, uncalibrated. No IMU; recorded TF used.',xref='paper',yref='paper',x=0,y=1.06,showarrow=False)])
    fig.write_html(args.output/'terrain_replay_3d.html',include_plotlyjs=True,auto_play=False)
    # Compact data for a standalone raster preview without requiring a browser.
    chosen=int(np.argmax([len(s) for s in surfaces]))
    np.savez_compressed(args.output/'preview_data.npz',cloud=np.asarray(clouds[chosen]),surface=np.asarray(surfaces[chosen]),colors=np.asarray(colors[chosen]))
    print('Saved',args.output/'terrain_replay_3d.html')


if __name__=='__main__': main()
