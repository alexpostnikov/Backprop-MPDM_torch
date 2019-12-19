#!/usr/bin/python3
import rospy
import random
from Visualizer import Visualizer
from Visualizer2 import Visualizer2
import numpy as np

import torch
from forward import calc_forces, pose_propagation, calc_cost_function
from Param import Param


if __name__ == '__main__':
    node = rospy.init_node('mpdm')
    param = Param()
    # visualisation staff
    vis_peds = Visualizer2('/peds')
    vis_robot = Visualizer2('/robot', color=1)
    vis_goals_ped = Visualizer2('/goals/ped',          size=[0.4, 0.4, 1.0])
    vis_goals_rob = Visualizer2('/goals/rob', color=1, size=[0.4, 0.4, 1.0])
    # visualisation staff

    # random env
    input_state = param.input_state
    goal = param.goal
    cost = torch.zeros(param.num_ped, 2)
    robot_pose = param.robot_init_pose
    robot_init_pose = param.robot_init_pose
    robot_goal = param.robot_goal

    while not rospy.is_shutdown():
        rf, af = calc_forces(input_state, goal, param.pedestrians_speed,
                             param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
        F = rf + af
        input_state = pose_propagation(
            F, input_state, param.DT, param.pedestrians_speed)
        cost += calc_cost_function(param.a, param.b, param.e,
                                   robot_goal, robot_init_pose, input_state[:, 0:2])
        goal = param.generate_new_goal(goal, input_state)

        # visualisation staff
        vis_peds.publish(torch.cat((input_state[1:], rf[1:], af[1:]), 1))
        vis_robot.publish(torch.cat((input_state[:1], rf[:1], af[:1]), 1))
        vis_goals_ped.publish(goal[1:])
        vis_goals_rob.publish(goal[0:1])
        # visualisation staff
        rospy.sleep(param.loop_sleep)
