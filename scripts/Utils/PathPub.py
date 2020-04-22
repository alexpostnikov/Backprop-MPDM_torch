import rospy

from nav_msgs.msg import Path

class PathPub:
    def __init__(self, topic_name="/path"):
        self.pub_path = rospy.Publisher(topic_name, Path,queue_size=0)

    def publish(self, msg):
        if msg == None:
            return
        self.pub_path.publish(msg)