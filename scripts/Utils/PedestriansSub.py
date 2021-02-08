import rospy
from mpdm.msg import Peds
from Utils.Utils import q2yaw
import numpy as np


class PedestriansSub:
    def __init__(self, topic="mpdm/peds"):
        self.peds = None
        self.goals = None
        self.previos_peds = None
        self.previos_goals = None
        self.peds_sub = rospy.Subscriber(
            topic, Peds, self.callback, queue_size=1)

    def callback(self, msg):
        # convert ros msg into input array for mpdm
        peds = []
        goals = []
        for ped in msg.peds:
            if ped.id.data is "0" or ped.id.data is "robot":
                continue
            x = ped.position.position.x
            y = ped.position.position.y
            yaw = q2yaw(ped.position.orientation)
            vx = ped.velocity.position.x
            vy = ped.velocity.position.y
            vyaw = q2yaw(ped.velocity.orientation)
            gx = ped.goal.position.x
            gy = ped.goal.position.y
            gyaw = q2yaw(ped.goal.orientation)
            peds.append([x, y, yaw, vx, vy, vyaw])
            goals.append([gx, gy, gyaw])
        self.peds = np.array(peds)
        self.goals = np.array(goals)
        self.new_msg = True

    def get_peds_state(self):
        try:
            self.new_msg = False
            self.previos_peds = self.peds.copy()
            self.previos_goals = self.goals.copy()
            return self.peds.copy(), self.goals.copy()
        except:
            return None, None
    def new_data(self):
        return (self.previos_peds != self.peds).any() or (self.previos_goals != self.goals).any()