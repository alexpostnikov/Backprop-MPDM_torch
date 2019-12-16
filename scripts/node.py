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
    vis_goals = Visualizer2('/goals')
    # visualisation staff

    # random env
    input_state = param.input_state
    goal = param.goal
    cost = torch.zeros(param.num_ped, 2)

    while not rospy.is_shutdown():
        F_attr = calc_forces(input_state, goal)
        input_state = pose_propagation(
            F_attr, input_state, param.DT, param.pedestrians_speed)
        cost += calc_cost_function(
            agents_pose=input_state[:, 0:2], a=param.a, b=param.b, e=param.e, robot_speed=param.robot_speed)
        goal = param.generate_new_goal(goal, input_state)

        # publish positions
        vis_peds.publish(input_state)
        vis_goals.publish(goal)
        rospy.sleep(param.loop_sleep)
