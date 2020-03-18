import rospy
from geometry_msgs.msg import PoseArray
# from tf.transformations import euler_from_quaternion
import numpy as np


class PedestriansSub:
    def __init__(self, topic="/peds/pose_vel_goal"):
        self.peds = None
        self.goals = None
        self.peds_sub = rospy.Subscriber(
            topic, PoseArray, self.callback, queue_size=1)

    def callback(self, msg):
        peds = []
        goals = []
        for i in range(0,len(msg.poses),3):
            x = msg.poses[i].position.x
            y = msg.poses[i].position.y
            quaternion = (
                msg.poses[i].orientation.x,
                msg.poses[i].orientation.y,
                msg.poses[i].orientation.z,
                msg.poses[i].orientation.w)
            # yaw = euler_from_quaternion(quaternion)[2]
            yaw = 0  # TODO: fix from tf.transformations import euler_from_quaternion
            vx = msg.poses[i+1].position.x
            vy = msg.poses[i+1].position.y
            quaternion = (
                msg.poses[i+1].orientation.x,
                msg.poses[i+1].orientation.y,
                msg.poses[i+1].orientation.z,
                msg.poses[i+1].orientation.w)
            # vyaw = euler_from_quaternion(quaternion)[2]
            vyaw = 0  # TODO: fix from tf.transformations import euler_from_quaternion
            peds.append([x, y, yaw, vx, vy, vyaw])
            gx = msg.poses[i+2].position.x
            gy = msg.poses[i+2].position.y
            quaternion = (
                msg.poses[i+2].orientation.x,
                msg.poses[i+2].orientation.y,
                msg.poses[i+2].orientation.z,
                msg.poses[i+2].orientation.w)
            # yaw = euler_from_quaternion(quaternion)[2]
            gyaw = 0  # TODO: fix from tf.transformations import euler_from_quaternion
            goals.append([gx,gy,gyaw])
        self.peds = np.array(peds)
        self.goals = np.array(goals)

    def get_peds_state(self):
        try:
            return self.peds.copy(), self.goals.copy()
        except:
            return None, None
