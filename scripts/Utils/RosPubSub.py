from Utils.RvizSub import RvizSub
from Utils.RobotPoseSub import RobotPoseSub
from Utils.MapSub import MapSub
from Utils.PathPub import PathPub
from Utils.PathSub import PathSub
from Utils.PedestriansSub import PedestriansSub

class RosPubSub:
    def __init__(self):
        self.map = MapSub()
        self.path = PathPub()
        self.rviz = RvizSub()
        self.inpath = PathSub()
        self.robot = RobotPoseSub()
        self.peds = PedestriansSub()

