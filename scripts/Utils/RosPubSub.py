from Utils.RvizSub import RvizSub
from Utils.RobotStateSub import RobotStateSub
from Utils.MapSub import MapSub
from Utils.PathPub import PathPub
from Utils.PedestriansSub import PedestriansSub

class RosPubSub:
    def __init__(self):
        self.map = MapSub()
        self.path = PathPub()
        self.rviz = RvizSub()
        self.robot = RobotStateSub()
        self.peds = PedestriansSub()

