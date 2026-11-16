import rclpy,time,json,yaml,pathlib,hashlib,sys
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import Image,CameraInfo,JointState
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data
rclpy.init();n=rclpy.create_node('isolated_startup_validation');clocks=[];images=[];cal=[];joints={}
subs=[]
subs.append(n.create_subscription(Clock,'/clock',lambda m:clocks.append(m.clock.sec+m.clock.nanosec/1e9),10))
subs.append(n.create_subscription(Image,'/right_gripper_camera/image_raw',lambda m:images.append((m.header.stamp.sec+m.header.stamp.nanosec/1e9,m.width,m.height,m.encoding)),qos_profile_sensor_data))
subs.append(n.create_subscription(CameraInfo,'/right_gripper_camera/camera_info',lambda m:cal.append((m.width,m.height,list(m.k))),qos_profile_sensor_data))
subs.append(n.create_subscription(JointState,'/joint_states',lambda m:joints.update(zip(m.name,m.position)),qos_profile_sensor_data))
controllers={}
for side in ['left','right']:
 c=n.create_client(ListControllers,'/'+side+'_controller_manager/list_controllers');assert c.wait_for_service(timeout_sec=4),side+' manager missing'
 response=None
 for attempt in range(2):
  f=c.call_async(ListControllers.Request());rclpy.spin_until_future_complete(n,f,timeout_sec=5)
  if f.done() and not f.cancelled():response=f.result();break
  f.cancel();print(json.dumps({'event':'controller_status_timeout','side':side,'attempt':attempt+1}),flush=True)
 assert response is not None,side+' manager response timeout after two bounded status queries'
 controllers[side]={x.name:x.state for x in response.controller};n.destroy_client(c)
end=time.monotonic()+6
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
expected=yaml.safe_load(pathlib.Path('/tmp/scene_right_initial.yaml').read_text());expected=expected.get('initial_positions',expected)
# The source map uses unprefixed names; messages use right_ names.
errors={k:abs(joints.get(k if k.startswith('right_') else 'right_'+k,999)-v) for k,v in expected.items()}
active=all(controllers[s].get(s+'_'+name)=='active' for s in ['left','right'] for name in ['joint_state_broadcaster','joint_group_velocity_controller','hande_position_controller'])
report={'controllers':controllers,'all_six_active':active,'clock_messages':len(clocks),'clock_advance':clocks[-1]-clocks[0] if len(clocks)>1 else 0,'image_messages':len(images),'image_dimensions':list(images[-1][1:]) if images else None,'camera_info':cal[-1] if cal else None,'joint_errors_rad':errors,'max_home_error_rad':max(errors.values()),'timestamp':time.time()}
report['passed']=active and report['clock_advance']>.1 and len(images)>5 and bool(cal) and max(errors.values())<.005 and images[-1][1:3]==(640,480) and abs(cal[-1][2][0]-184.752)<.05
report['passed']=bool(report['passed']);print(json.dumps(report,indent=2));n.destroy_node();rclpy.shutdown();sys.exit(0 if report['passed'] else 1)
