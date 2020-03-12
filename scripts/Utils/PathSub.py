import rospy

from nav_msgs.msg import Path

class PathSub:
    def __init__(self, topic_name="/global_path"):
        self.last_path = Path()
        self.pub_path = rospy.Subscriber(topic_name, Path, self.callback, queue_size=2)
    
    def callback(self, msg):
        self.last_path = msg