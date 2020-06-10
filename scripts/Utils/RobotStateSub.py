#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, PoseStamped
from nav_msgs.msg import Path
from Utils.Utils import q2yaw
import numpy as np


class RobotStateSub:
    def __init__(self, topic_pose="/odom", topic_path="/move_base/GlobalPlanner/plan", topic_PoseStamped='/move_base_simple/goal', topic_PointStamped="/clicked_point", topic_PoseWithCovarianceStamped="/initialpose"):
        # robot = [x, y, yaw, speed_x, speed_y, speed_yaw]
        self.pose = np.zeros(3)
        self.speed = np.zeros(3)
        self.robot = np.zeros(6)
        self.robot[:3] = None, None, None
        self.path = None
        self.goal = np.zeros(3)
        self.goal[:] = None, None, None
        self.sub_path = rospy.Subscriber(
            topic_path, Path, self.callback_path, queue_size=1)
        self.sub_pose = rospy.Subscriber(
            topic_pose, PoseStamped, self.callback_pose, queue_size=1)
        # below subs from rviz topics
        self.sub_PoseStamped = rospy.Subscriber(
            topic_PoseStamped, PoseStamped, self.callback_goal, queue_size=1)
        self.sub_PointStamped = rospy.Subscriber(
            topic_PointStamped, PointStamped, self.callback_goal, queue_size=1)
        self.sub_PoseWithCovarianceStamped = rospy.Subscriber(
            topic_PoseWithCovarianceStamped, PoseWithCovarianceStamped, self.callback_initialpose, queue_size=1)

    def callback_goal(self, msg):
        if 'geometry_msgs/PoseStamped' in msg._type:
            self.goal[0] = msg.pose.position.x
            self.goal[1] = msg.pose.position.y
            self.goal[2] = q2yaw(msg.pose.orientation)
        if 'geometry_msgs/PointStamped' in msg._type:
            self.goal[0] = msg.point.x
            self.goal[1] = msg.point.y
            self.goal[2] = 0
        
    def callback_initialpose(self, msg):
        self.pose[0] = msg.pose.pose.position.x
        self.pose[1] = msg.pose.pose.position.y
        self.pose[2] = q2yaw(msg.pose.pose.orientation)
    

    def callback_path(self, msg):
        self.path = []
        for p in msg.poses:
            yaw = q2yaw(p.pose.orientation)
            point = [p.pose.position.x, p.pose.position.y, yaw]
            self.path.append(point)
        self.path = np.array(self.path)
        self.goal = self.path[-1]


    def callback_pose(self, msg):
        yaw = q2yaw(msg.pose.orientation)
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.pose[:] = x, y, yaw
        self.robot[:3] = self.pose[:]

    def get_robot_state(self):
        if np.isnan(self.robot)[0] or np.isnan(self.robot)[1] or np.isnan(self.robot)[2]:
            return None, None
            # if goal is None, just copy current position
        if np.isnan(self.goal)[0] or np.isnan(self.goal)[1] or np.isnan(self.goal)[2]:
            self.goal = self.pose.copy()
        return self.robot.copy(), self.goal.copy()
