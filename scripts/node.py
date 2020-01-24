#!/usr/bin/python3
import rospy
import random

from Visualizer2 import Visualizer2, Rviz_sub 
import numpy as np

import torch
from forward import calc_forces, pose_propagation, calc_cost_function
from Param import Param
from optimization import  get_poses_probability, Linear, optimize, lr, rotate, come_to_me
# from optimisation import Linear
import torch.nn as nn
import time
from utils import check_poses_not_the_same, setup_logger
import math

def apply_policy(policy, state, goals, robot_speed):
    if policy == "stop":
        # starting_poses[0,2:4] = -20 * starting_poses[0,2:4].clone()
        robot_speed = 0.0001
        # return state, goals.clone(), robot_speed
    if policy == "right":
        Gx, Gy = rotate([starting_poses[0,0].data, starting_poses[0,1].data], [goals[0,0].data, goals[0,1].data ], -math.pi/4.)
        Gx, Gy = come_to_me([starting_poses[0,0].data, starting_poses[0,1].data],[Gx, Gy ],0.5)
        # with torch.no_grad():
        goals[0,0:2] =  torch.tensor(([Gx, Gy ]))
        robot_speed = 1.0
        
        # goals[0,0:2] = torch.tensor(([Gx, Gy ]))
        # return state, goals, robot_speed
    if policy == "left":
        Gx, Gy = rotate([starting_poses[0,0].data, starting_poses[0,1].data], [goals[0,0].data, goals[0,1].data ], math.pi/4.)
        Gx, Gy = come_to_me([starting_poses[0,0].data, starting_poses[0,1].data],[Gx, Gy ],0.5)
        # with torch.no_grad():
        goals[0,0:2] =  torch.tensor(([Gx, Gy ]))
        robot_speed = 1.0
        # return state, goals, robot_speed

    if policy == "fast":
        robot_speed = 2.0
        # return state, goals.clone(), robot_speed

    if policy == "slow":
        robot_speed = 0.5
        # return state, goals.clone(), robot_speed

    if policy == "go-solo":
        robot_speed = 1.0
    return state, goals, robot_speed

    


if __name__ == '__main__':
      # torch.autograd.set_detect_anomaly(True)
    param = Param()    
    rospy.init_node("mpdm")
    ##### logging init   #####
    if param.do_logging:        
        logger = setup_logger(logger_name = "mpdm node")
        logger.info("sim params: num_ped" + str(param.num_ped) + "   num of layers " + str(param.number_of_layers))
    
    if param.do_visualization:

        pedestrians_visualizer = Visualizer2("peds", starting_id=1)
        initial_pedestrians_visualizer = Visualizer2("peds_initial", color=3, size=[0.6/3, 0.6/3, 1.8/3], with_text = False)
        ped_goals_visualizer = Visualizer2("goals/ped",size=[0.1, 0.1, 0.5])
        initial_ped_goals_visualizer = Visualizer2("goals/init_ped", size=[0.05, 0.05, 0.25], color=3, with_text = True)
        robot_visualizer = Visualizer2("robot", color=1, with_text = False)
        policy_visualizer = Visualizer2("robot_policy", color=1 ,  with_text = False)
        learning_vis = Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2, with_text = False)

    modules = []
    for i in range (0, param.number_of_layers):
        modules.append(Linear())

    manual_goal = [None, None]
    manual_pose = [None, None]
    goal_sub = Rviz_sub(manual_goal, manual_pose)
    
    sequential = nn.Sequential(*modules)

    for i in range (10):
        if rospy.is_shutdown():
                break
        observed_state = param.input_state.clone().detach()

        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
        robot_init_pose = observed_state[0,0:2]#param.robot_init_pose.requires_grad_(True) 
        goals = param.goal.clone().detach().requires_grad_(False)
        
        # ##### MODEL CREATING ######
        modules = []
        for i in range (0, param.number_of_layers):
            modules.append(Linear())

        sequential = nn.Sequential(*modules)
        # #### OPTIMIZATION ####
        global_start_time = time.time()

        policies = ["stop", "go-solo", "left", "right"]#, "slow"]#, "fast", , "left", "right"]
        # policies = ["stop", "go-solo"]
        results = []
        for policy in policies:
            starting_poses = observed_state.clone()
            starting_poses, goals_in_policy, param.robot_speed = apply_policy(policy, starting_poses, goals.clone(), param.robot_speed)
            goals_in_policy.requires_grad_(True)
            cost = optimize(param.optim_epochs, sequential, starting_poses, 
                param.robot_init_pose, param, 
                goals_in_policy ,lr,ped_goals_visualizer, 
                initial_pedestrians_visualizer, 
                pedestrians_visualizer, robot_visualizer, 
                learning_vis, initial_ped_goals_visualizer, policy)
                
            results.append(cost)
        
        print (np.array(results) - min(results))
        pol_numb = results.index(min(results))
        policy = policies[pol_numb]
        print (policy)
        
        with torch.no_grad():
            param.input_state, goals, param.robot_speed = apply_policy(policy, param.input_state, goals.clone(), param.robot_speed)
        
        
        with torch.no_grad():
            # propagate
            rf, af = calc_forces(param.input_state, goals, param.pedestrians_speed, param.robot_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
            F = rf + af
            observed_state_new = pose_propagation(F, param.input_state, param.DT, param.pedestrians_speed, param.robot_speed)

            #update scene with new poses & check is goals achieved
            param.update_scene(observed_state_new, param.goal)
            goals = param.goal.clone()
            goals = param.generate_new_goal(goals, param.input_state)

            if manual_goal[0] is not None:
                param.goal[0,0] = manual_goal[0]
                param.goal[0,1] = manual_goal[1]
                manual_goal[0] = None
            
            if manual_pose[0] is not None:
                param.input_state[0,0] = manual_pose[0]
                param.input_state[0,1] = manual_pose[1]

                # param.robot_init_pose[0] = manual_pose[0]
                # param.robot_init_pose[1] = manual_pose[1]
                manual_pose[0] = None
            
            param.robot_init_pose = param.input_state[0,0:2]
                
            
            goals.requires_grad_(True)
            # print("goals",goals)
        policy_visualizer.publish(param.input_state[0:1],  policy)