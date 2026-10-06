"""Verify the review topic and capture only the RViz window."""
import json
import subprocess
import time
from pathlib import Path
import rclpy
from std_msgs.msg import String
from PyQt5.QtWidgets import QApplication

rclpy.init()
node=rclpy.create_node('rviz_review_check')
received=[]
node.create_subscription(String,'/traversability/terrain_status',lambda m:received.append(json.loads(m.data)),10)
deadline=time.monotonic()+12
while time.monotonic()<deadline and not any('forward' in item for item in received):
    rclpy.spin_once(node,timeout_sec=.2)
print('TERRAIN_STATUS',json.dumps(received[-1] if received else {},ensure_ascii=False),flush=True)
print('NODES',node.get_node_names(),flush=True)
node.destroy_node();rclpy.shutdown()
windows=subprocess.check_output(['xdotool','search','--name','terrain_review']).decode().split()
app=QApplication([])
if windows:
    window=windows[-1]
    subprocess.run(['xdotool','windowactivate','--sync',window],check=True,timeout=5)
    app.processEvents()
    path=Path(__file__).resolve().parents[1]/'test_results/terrain_3d/rviz_live.png'
    print('SCREENSHOT_SAVED',app.primaryScreen().grabWindow(int(window)).save(str(path)))
else:
    raise RuntimeError('RViz review window was not found')
