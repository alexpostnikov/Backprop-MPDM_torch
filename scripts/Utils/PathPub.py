import rospy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from Utils.Utils import yaw2q


class PathPub:
    def __init__(self, topic_name="/path"):
        self.pub_path = rospy.Publisher(topic_name, Path, queue_size=0)

    def publish(self, msg):
        if msg is None:
            return
        self.pub_path.publish(msg)

    def publish_from_array(self, array, frame_id="map"):
        ros_msg = self.array_to_ros_path(array, frame_id)
        self.publish(ros_msg)

    def publish_from_tensor(self, array, frame_id="map"):
        ros_msg = self.array_to_ros_path(array.detach().numpy(), frame_id)
        self.publish(ros_msg)

    def array_to_ros_path(self, array, frame_id):
        path = Path()
        path.header.frame_id = frame_id
        if array is not None:
            for pose in array:
                path.poses.append(self.ps(pose[0], pose[1], pose[2], frame_id))
        return path

    def ps(self, x, y, yaw, frame_id):
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation = yaw2q(yaw)
        return ps
