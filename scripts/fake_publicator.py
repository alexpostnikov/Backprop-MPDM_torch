#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Twist
from nav_msgs.msg import Path

# from tf.transformations import euler_from_quaternion, quartenion_from_euler

def ps(x,y,yaw=0,frame = "map"):
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.orientation.w = 1
    return ps

def p(x,y,yaw=0,frame = "map"):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.orientation.w = 1
    return p

def t(x,y,yaw=0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z =yaw
    return t
    
if __name__ == '__main__':
    rospy.init_node("fake_publicator")
    frame = "map"
    # position
    robot_pose_pub = rospy.Publisher("/odom", PoseStamped, queue_size=1)
    robot_pose = ps(1,2)
    # velocity
    robot_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    robot_vel = t(1,0)
    # path(goal)
    robot_path_pub = rospy.Publisher("/move_base/GlobalPlanner/plan", Path, queue_size=1)
    path = Path()
    path.header.frame_id = frame
    path.poses.append(ps(1,2))
    path.poses.append(ps(2.5,2.5))
    path.poses.append(ps(5,2))
    # pedestrians position_velocity_goal
    ped_pub = rospy.Publisher("/peds/pose_vel_goal", PoseArray, queue_size=1)
    peds = PoseArray()
    peds.header.frame_id = frame
    # 1
    # pose[x=2,y=2,yaw=0] 
    # vel[vx=0.5,vy=0,vyaw=0] 
    # goal[x=0,y=0,yaw=0]
    peds.poses.append(p(2,2))
    peds.poses.append(p(0.5,0))
    peds.poses.append(p(3,3))
    # 2
    peds.poses.append(p(2,1))
    peds.poses.append(p(0.5,0))
    peds.poses.append(p(0,0))
    # 3
    peds.poses.append(p(1.5,1.5))
    peds.poses.append(p(0.5,0))
    peds.poses.append(p(5,3))


    while not (rospy.is_shutdown()):
        robot_pose_pub.publish(robot_pose)
        robot_vel_pub.publish(robot_vel)
        robot_path_pub.publish(path)
        ped_pub.publish(peds)
        rospy.sleep(0.5)

    # MPDM
