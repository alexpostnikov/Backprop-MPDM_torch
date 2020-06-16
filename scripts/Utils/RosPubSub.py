from Utils.RobotStateSub import RobotStateSub
from Utils.MapSub import MapSub
from Utils.PathPub import PathPub
from Utils.PedestriansSub import PedestriansSub
from Utils.LearningPub import LearningPub

class RosPubSub:
    def __init__(self):
        new_data_available = False
        self.map = MapSub()
        self.robot = RobotStateSub()
        self.peds = PedestriansSub()
        self.path = PathPub()
        self.learning = LearningPub()
    
    def new_data_available(self):
        # check data
        return self.robot.new_data() or self.peds.new_data() 

