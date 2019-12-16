#!/usr/bin/python3
import rospy
import random
from Visualizer import Visualizer
from Visualizer2 import Visualizer2
import numpy as np

import torch
from forward import calc_forces, pose_propagation, is_goal_achieved, calc_cost_function
from force_attract_with_dest import generate_new_goal


if __name__ == '__main__':
    node = rospy.init_node('mpdm')
    loop_rate = 10.

    loop_sleep = 1/loop_rate
    # visualisation staff
    vis_peds = Visualizer2('/peds')
    vis_goals = Visualizer2('/goals')
    # visualisation staff

    # random env
    num_ped = 4
    # , [1.0,2.5,0.0,0.0]) )
    # input_state = torch.tensor(([2.0, 0.5, 0.0, -1.], [2.0, 0.5, 0.0, -1.]))
    input_state = 4*torch.rand((num_ped,4))
    k = 2.2
    DT = 0.2
    pedestrians_speed = 0.5
    input_state = input_state.view(-1, 4)
    # goal = torch.tensor(([4.0, 1.0], [4.0, 1.0]))
    goal = 4*torch.rand((num_ped,2))
    goal = goal.view(-1, 2)
    cost = torch.zeros(num_ped,2)


    while not rospy.is_shutdown():
        F_attr = calc_forces(input_state, goal)
        input_state = pose_propagation(F_attr, input_state)
        cost+=calc_cost_function(agents_pose=input_state[:,0:2])

        res = is_goal_achieved(input_state, goal)
        if any(res) == True:
            goal = generate_new_goal(goal, res, input_state, 4)
        # publish positions
        vis_peds.publish(input_state)
        vis_goals.publish(goal)
        rospy.sleep(loop_sleep)
       
       
       




