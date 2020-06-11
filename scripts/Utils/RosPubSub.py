from Utils.RobotStateSub import RobotStateSub
from Utils.MapSub import MapSub
from Utils.PathPub import PathPub
from Utils.PedestriansSub import PedestriansSub
from Utils.LearningPub import LearningPub

class RosPubSub:
    def __init__(self):
        self.map = MapSub()
        self.robot = RobotStateSub()
        self.peds = PedestriansSub()
        self.path = PathPub()
        self.learning = LearningPub()

