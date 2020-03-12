import rospy

from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_quaternion
import numpy as np


class RobotPoseSub:
    def __init__(self, topic_pose="/odometry", topic_speed="/twist"):
        # robot = [x, y, yaw, speed_x, speed_y, speed_yaw]
        self.pose = [0, 0, 0]
        self.speed = [0, 0, 0]
        self.robot = np.zeros(6)
        self.sub_pose = rospy.Subscriber(
            topic_pose, PoseStamped, self.callback_pose, queue_size=1)
        self.sub_speed = rospy.Subscriber(
            topic_speed, Twist, self.callback_twist, queue_size=1)

    def callback_pose(self, msg):
        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w)
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = euler_from_quaternion(quaternion)[2]
        self.pose[:] = x, y, yaw
        self.robot[:3] = self.pose[:]

    def callback_twist(self, msg):
        x = msg.linear.x
        y = msg.linear.y
        yaw = msg.angular.z
        self.speed[:] = x, y, yaw
        self.robot[3:] = self.speed[:]

    def get_robot_state(self):
        return self.robot.copy()
