import rospy
from geometry_msgs.msg import PoseArray
from Utils.Utils import quaternion_to_euler
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
        for i in range(0, len(msg.poses), 3):
            x = msg.poses[i].position.x
            y = msg.poses[i].position.y
            _, _, yaw = quaternion_to_euler(
                msg.poses[i].orientation.x,
                msg.poses[i].orientation.y,
                msg.poses[i].orientation.z,
                msg.poses[i].orientation.w)
            vx = msg.poses[i+1].position.x
            vy = msg.poses[i+1].position.y
            _, _, vyaw = quaternion_to_euler(
                msg.poses[i+1].orientation.x,
                msg.poses[i+1].orientation.y,
                msg.poses[i+1].orientation.z,
                msg.poses[i+1].orientation.w)
            peds.append([x, y, yaw, vx, vy, vyaw])
            gx = msg.poses[i+2].position.x
            gy = msg.poses[i+2].position.y
            _, _, gyaw = quaternion_to_euler(
                msg.poses[i+2].orientation.x,
                msg.poses[i+2].orientation.y,
                msg.poses[i+2].orientation.z,
                msg.poses[i+2].orientation.w)
            goals.append([gx, gy, gyaw])
        self.peds = np.array(peds)
        self.goals = np.array(goals)

    def get_peds_state(self):
        try:
            return self.peds.copy(), self.goals.copy()
        except:
            return None, None
