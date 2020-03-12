import rospy
from geometry_msgs.msg import PoseArray
from tf.transformations import euler_from_quaternion
import numpy as np


class PedestriansSub:
    def __init__(self, topic="/pedestrians"):
        self.peds = np.zeros([6, 3])
        self.peds_sub = rospy.Subscriber(
            topic, PoseArray, self.callback, queue_size=1)

    def callback(self, msg):
        peds = []
        for p in msg.poses:
            x = p.position.x
            y = p.position.y
            quaternion = (
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w)
            yaw = euler_from_quaternion(quaternion)[2]
            peds.append([x, y, yaw, 0, 0, 0])
        self.peds = np.array(peds)

    def get_peds_state(self):
        return self.peds.copy()
