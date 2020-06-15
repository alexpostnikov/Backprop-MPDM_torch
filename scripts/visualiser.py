#!/usr/bin/python3
import rospy
from mpdm.msg import Learning
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Pose, Vector3, PointStamped, PoseStamped, PoseWithCovarianceStamped, Quaternion
from std_msgs.msg import ColorRGBA
import numpy as np
from copy import deepcopy
import math

class Visualiser5:
    def __init__(self, frame="map", with_text=False,):
        self.with_text = with_text
        self.frame = frame

        self.agent_scale = Vector3(0.3, 0.6, 1.8)
        self.goal_scale = Vector3(0.3, 0.6, 0.5)
        self.propagation_scale = Vector3(0.15, 0.3, 0.8)
        self.text_scale = Vector3(0, 0, 0.5)
        self.first_arrow_scale = Vector3(0.008, 0.02, 0.02)
        self.arrow_scale = Vector3(0.02, 0.1, 0.1)

        self.robot_color = ColorRGBA(0, 0.9, 0, 1)  # - green
        self.robot_goal_color = ColorRGBA(0, 1, 0, 1)  # - green
        self.ped_color = ColorRGBA(0.9, 0.9, 0.9, 1)  # - white
        self.ped_goal_color = ColorRGBA(1, 1, 1, 1)  # - white
        self.text_color = ColorRGBA(0, 0, 0, 1)  # - black
        self.propagation_color = ColorRGBA(0, 0, 1, 0.5)  # - blue
        self.covariance_color = ColorRGBA(0, 1, 0, 0.5)  # - green

        self.arrow_colors = [
            ColorRGBA(0, 1, 0, 1),    # force 1 - green
            ColorRGBA(0, 0, 1, 1),    # force 2 - blue
            ColorRGBA(1, 0, 0, 1),    # force 3 - red
            ColorRGBA(0, 0, 0, 1)     # force 4 - black
        ]

        self.pub_peds = rospy.Publisher(
            "mpdm/vis/peds", MarkerArray, queue_size=1)
        self.pub_peds_goals = rospy.Publisher(
            "mpdm/vis/peds_goals", MarkerArray, queue_size=1)
        self.pub_robot = rospy.Publisher(
            "mpdm/vis/robot", MarkerArray, queue_size=1)
        self.pub_robot_goal = rospy.Publisher(
            "mpdm/vis/robot_goal", MarkerArray, queue_size=1)
        self.pub_forces = rospy.Publisher(
            "mpdm/vis/forces", MarkerArray, queue_size=1)
        self.pub_propagation = rospy.Publisher(
            "mpdm/vis/propagation", MarkerArray, queue_size=1)
        self.pub_covariances = rospy.Publisher(
            "mpdm/vis/covariances", MarkerArray, queue_size=1)
        self.pub_learning = rospy.Publisher(
            "mpdm/vis/learning", MarkerArray, queue_size=1)
        self.sub_learning = rospy.Subscriber(
            "mpdm/debug", Learning, self.callback_learning, queue_size=1)

    def callback_learning(self, msg):
        peds_msg = MarkerArray()
        peds_goals_msg = MarkerArray()
        robot_msg = MarkerArray()
        robot_goal_msg = MarkerArray()
        forces_msg = MarkerArray()
        propagation_msg = MarkerArray()
        covariances_msg = MarkerArray()
        learning_msg = MarkerArray()

        # founding out the best_epoch in learning
        best_epoch = msg.epochs[0]
        for epoch in msg.epochs:
            if best_epoch.cost.data > epoch.cost.data:
                best_epoch = epoch
        # founding out robot state and add to markers
        for agent in best_epoch.steps[0].peds:
            if agent.id.data is "0" or agent.id.data is "robot":
                robot = agent

        id = 0
        # ROBOT and ROBOT_GOAL
        robot_marker = Marker(
            id=id,
            type=Marker.SPHERE,
            action=Marker.ADD,
            scale=self.agent_scale,
            color=self.robot_color,
            pose=deepcopy(robot.position)
        )
        robot_marker.header.frame_id = self.frame
        robot_marker.pose.position.z = robot_marker.scale.z/2.
        robot_msg.markers.append(robot_marker)
        robot_goal_marker = Marker(
            id=id,
            type=Marker.CUBE,
            action=Marker.ADD,
            scale=self.goal_scale,
            color=self.robot_goal_color,
            pose=robot.goal
        )
        robot_goal_marker.header.frame_id = self.frame
        robot_goal_marker.pose.position.z = robot_goal_marker.scale.z/2.
        robot_goal_msg.markers.append(robot_goal_marker)
        # PEDS, PEDS GOALS
        for ped in best_epoch.steps[0].peds:
            if ped.id.data is "0" or ped.id.data is "robot":
                continue  # just skip the robot
            ped_marker = Marker(
                id=id,
                type=Marker.SPHERE,
                action=Marker.ADD,
                scale=self.agent_scale,
                color=self.ped_color,
                pose=deepcopy(ped.position)
            )
            ped_marker.header.frame_id = self.frame
            ped_marker.pose.position.z = ped_marker.scale.z/2.
            peds_msg.markers.append(ped_marker)
            ped_goal_marker = Marker(
                id=id,
                type=Marker.CUBE,
                action=Marker.ADD,
                scale=self.goal_scale,
                color=self.ped_goal_color,
                pose=ped.goal
            )
            ped_goal_marker.header.frame_id = self.frame
            ped_goal_marker.pose.position.z = ped_goal_marker.scale.z/2.
            peds_goals_msg.markers.append(ped_goal_marker)
            id += 1
        # FORCES
        # TODO: add this data into msgs
        # PROPAGATION and COVARIANCES
        id = 0
        for step in best_epoch.steps:
            for ped in step.peds:
                pped_marker = Marker(
                    id=id,
                    type=Marker.SPHERE,
                    action=Marker.ADD,
                    scale=self.propagation_scale,
                    color=self.propagation_color,
                    pose=deepcopy(ped.position)
                )
                pped_marker.header.frame_id = self.frame
                pped_marker.pose.position.z = pped_marker.scale.z/2.
                propagation_msg.markers.append(pped_marker)
                covariance_marker = Marker(
                    id=id,
                    type=Marker.SPHERE,
                    action=Marker.ADD,
                    scale=Vector3(0, 0, 0.05),
                    color=self.covariance_color,
                    pose=ped.position
                )
                covariance_marker.scale.x = ped.cov_pose.position.x
                covariance_marker.scale.y = ped.cov_pose.position.y
                if math.fabs(covariance_marker.scale.x) < 0.01 or math.fabs(covariance_marker.scale.y) < 0.01:
                    covariance_marker.scale.x = 0.01
                    covariance_marker.scale.y = 0.01
                covariance_marker.header.frame_id = self.frame
                covariance_marker.pose.position.z = covariance_marker.scale.z/2.
                covariances_msg.markers.append(covariance_marker)
                id += 1

        # PUBLISH ALL MSGS
        self.pub_peds.publish(peds_msg)
        self.pub_peds_goals.publish(peds_goals_msg)
        self.pub_robot.publish(robot_msg)
        self.pub_robot_goal.publish(robot_goal_msg)
        # self.pub_forces.publish(forces_msg)
        self.pub_propagation.publish(propagation_msg)
        self.pub_covariances.publish(covariances_msg)
        # self.pub_learning.publish(learning_msg)

    def yaw2q(self, yaw):
        return Quaternion(x=0, y=0, z=np.sin(yaw/2), w=np.cos(yaw/2))


if __name__ == '__main__':
    rospy.init_node("mpdm_visualiser")
    vis = Visualiser5()
    while not rospy.is_shutdown():
        rospy.sleep(0.05)
    exit()
