#!/usr/bin/python3
import rospy
from mpdm.msg import Learning
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import ColorRGBA
from copy import deepcopy
import math


class Visualiser5:
    def __init__(self, frame="map", with_text=False):
        self.with_text = with_text
        self.frame = frame

        self.agent_scale = Vector3(0.3, 0.6, 1.8)
        self.goal_scale = Vector3(0.2, 0.35, 0.2)
        self.propagation_scale = Vector3(0.15, 0.3, 0.8)
        self.learning_scale = Vector3(0.15, 0.3, 0.8)
        self.text_scale = Vector3(0, 0, 0.5)
        self.velocity_arrow_scale = Vector3(0.008, 0.02, 0.02)
        self.force_arrow_scale = Vector3(0.008, 0.02, 0.02)
        self.arrow_scale = Vector3(0.02, 0.1, 0.1)

        self.robot_color = ColorRGBA(0, 0.9, 0, 1)  # - green
        self.robot_goal_color = ColorRGBA(0, 1, 0, 1)  # - green
        self.ped_color = ColorRGBA(0.9, 0.9, 0.9, 1)  # - white
        self.ped_goal_color = ColorRGBA(1, 1, 1, 1)  # - white
        self.text_color = ColorRGBA(0, 0, 0, 1)  # - black
        self.propagation_color = ColorRGBA(0, 0, 1, 0.3)  # - blue
        self.learning_color = ColorRGBA(0, 0, 1, 0.1)  # - blue
        self.covariance_color = ColorRGBA(0, 1, 0, 0.5)  # - green
        self.velocity_arrow_color = ColorRGBA(0, 1, 0, 1)  # - green
        self.force_repulsive_arrow_color = ColorRGBA(1, 0.1, 0.1, 1)  # - red
        self.force_goal_arrow_color = ColorRGBA(0.1, 0.1, 1, 1)  # - blue
        self.force_wall_arrow_color = ColorRGBA(0.1, 1, 0.1, 1)  # - green

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
        # TODO: add text on agents and goals

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
        # velocity arrow
        robot_speed_marker = Marker(
            id=id+100,
            type=Marker.ARROW,
            action=Marker.ADD,
            scale=self.velocity_arrow_scale,
            color=self.velocity_arrow_color,
            points=[deepcopy(robot.position.position), self.p_summ(
                deepcopy(robot.position.position), robot.velocity.position)]
        )
        robot_speed_marker.header.frame_id = self.frame
        robot_speed_marker.pose.orientation.w = 1.
        robot_msg.markers.append(robot_speed_marker)

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

        # PEDS
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
            # velocity arrow
            id = id+1000
            ped_speed_marker = Marker(
                id=id,
                type=Marker.ARROW,
                action=Marker.ADD,
                scale=self.velocity_arrow_scale,
                color=self.velocity_arrow_color,
                points=[deepcopy(ped.position.position), self.p_summ(
                    deepcopy(ped.position.position), ped.velocity.position)]
            )
            ped_speed_marker.header.frame_id = self.frame
            ped_speed_marker.pose.orientation.w = 1.
            peds_msg.markers.append(ped_speed_marker)
           # goal markers
            id = id+1000
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
                # velocity arrow
                small_ = deepcopy(ped.velocity.position)
                small_.x *= 0.5
                small_.y *= 0.5
                small_.z *= 0.5
                pped_speed_marker = Marker(
                    id=id+100,
                    type=Marker.ARROW,
                    action=Marker.ADD,
                    scale=self.velocity_arrow_scale,
                    color=self.velocity_arrow_color,
                    points=[deepcopy(ped.position.position), self.p_summ(
                        deepcopy(ped.position.position), small_)]
                )
                pped_speed_marker.pose.orientation.w = 1.
                pped_speed_marker.header.frame_id = self.frame

                propagation_msg.markers.append(pped_speed_marker)

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
        id = 0
        for epoch in msg.epochs:
            for step in epoch.steps:
                for ped in step.peds:
                    # LEARNING
                    lped_marker = Marker(
                        id=id,
                        type=Marker.SPHERE,
                        action=Marker.ADD,
                        scale=self.learning_scale,
                        color=self.learning_color,
                        pose=deepcopy(ped.position)
                    )
                    lped_marker.header.frame_id = self.frame
                    lped_marker.pose.position.z = lped_marker.scale.z/2.
                    learning_msg.markers.append(lped_marker)
                    id += 1
                    # FORCES
                    force_multiplicator = 0.1
                    force_multiplicator_debug = 0.01
                    # ped.force_repulsive
                    id = id+1000
                    ped_force_repulsive_marker = Marker(
                        id=id,
                        type=Marker.ARROW,
                        action=Marker.ADD,
                        scale=self.force_arrow_scale,
                        color=self.force_repulsive_arrow_color,
                        points=[deepcopy(ped.position.position), self.p_summ(
                            deepcopy(ped.position.position), ped.force_repulsive.position,force_multiplicator_debug)]
                    )
                    ped_force_repulsive_marker.header.frame_id = self.frame
                    ped_force_repulsive_marker.pose.orientation.w = 1.
                    forces_msg.markers.append(ped_force_repulsive_marker)
                    # ped.force_goal
                    id = id+1000
                    ped_force_goal_marker = Marker(
                        id=id,
                        type=Marker.ARROW,
                        action=Marker.ADD,
                        scale=self.force_arrow_scale,
                        color=self.force_goal_arrow_color,
                        points=[deepcopy(ped.position.position), self.p_summ(
                            deepcopy(ped.position.position), ped.force_goal.position,force_multiplicator_debug)]
                    )
                    ped_force_goal_marker.header.frame_id = self.frame
                    ped_force_goal_marker.pose.orientation.w = 1.
                    forces_msg.markers.append(ped_force_goal_marker)
                    # ped.force_wall
                    id = id+1000
                    ped_force_wall_marker = Marker(
                        id=id,
                        type=Marker.ARROW,
                        action=Marker.ADD,
                        scale=self.force_arrow_scale,
                        color=self.force_wall_arrow_color,
                        points=[deepcopy(ped.position.position), self.p_summ(
                            deepcopy(ped.position.position), ped.force_wall.position,force_multiplicator)]
                    )
                    ped_force_wall_marker.header.frame_id = self.frame
                    ped_force_wall_marker.pose.orientation.w = 1.
                    forces_msg.markers.append(ped_force_wall_marker)
        
        
        # PUBLISH ALL MSGS
        self.pub_peds.publish(peds_msg)
        self.pub_peds_goals.publish(peds_goals_msg)
        self.pub_robot.publish(robot_msg)
        self.pub_robot_goal.publish(robot_goal_msg)
        self.pub_forces.publish(forces_msg)
        self.pub_propagation.publish(propagation_msg)
        self.pub_covariances.publish(covariances_msg)
        self.pub_learning.publish(learning_msg)

    def p_summ(self, p1, p2,k=1):
        return Point(x=p1.x+p2.x*k, y=p1.y+p2.y*k, z=p1.z+p2.z*k)


if __name__ == '__main__':
    rospy.init_node("mpdm_visualiser")
    vis = Visualiser5()
    while not rospy.is_shutdown():
        rospy.sleep(0.05)
    exit()
