#! /home/robot/miniconda3/envs/bonsai/bin/python

import sys
print (sys.version)
import torch
from force_attract_with_dest import force_goal, pose_propagation, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, calc_forces
from Visualizer2 import Visualizer2, Goal_sub
from Param import Param
import rospy
import time
from utils import check_poses_not_the_same
import torch.nn as nn
import torch.optim as optim
import numpy as np

lr = 5*10**-6


class Linear(nn.Module):
    def __init__(self):                     #input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.register_parameter('init_state', None)

    def forward(self, input):
        state, cost, stacked_trajectories_for_visualizer = input
        
        if self.init_state is None:
            self.init_state = nn.Parameter(state).requires_grad_(True)
            self.init_state.retain_grad()
            rf, af = calc_forces(self.init_state, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
            F = rf + af
            out = pose_propagation(F, self.init_state, param.DT, param.pedestrians_speed)
        else:
            if type(state) == type(nn.Parameter(torch.Tensor(1))):
                state = torch.tensor(state)
            rf, af = calc_forces(state, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
            F = rf + af
            out = pose_propagation(F, state, param.DT, param.pedestrians_speed)
        
        
        
        temp = calc_cost_function(param.a, param.b, param.e, goals, robot_init_pose, out, observed_state)
        cost = cost + temp.view(-1,1)
        stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer,inner_data.clone()))
        
        return (out, cost, stacked_trajectories_for_visualizer) 


if __name__ == '__main__':
    rospy.init_node("vis")
    # goal = [1.,1.]
    # Goal_sub(goal)
    # while not rospy.is_shutdown():
    #     # print(goal)
    #     time.sleep(0.1)

    pedestrians_visualizer = Visualizer2("peds")
    torch.manual_seed(1)
    param = Param()
    look_ahead_steps = int(param.look_ahead_seconds/ param.DT)
    torch.autograd.set_detect_anomaly(True)
    observed_state = param.input_state
    goals = param.goal
    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    robot_init_pose = observed_state[0,0:2]#param.robot_init_pose.requires_grad_(True)

    gradient = None
    ped_goals_visualizer = Visualizer2("goals/ped",size=[0.1, 0.1, 0.5])
    robot_visualizer = Visualizer2("robot", color=1)
    learning_vis= Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2, with_text = False)

    starting_poses = observed_state.clone()


    ##### MODEL CREATING ######
    modules = []
    for i in range (0, look_ahead_steps):
        modules.append(Linear())

    sequential = nn.Sequential(*modules)
    
    ped_goals_visualizer.publish(goals)
    
    # exit()
    inner_data = torch.nn.Parameter(starting_poses.clone())

    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    stacked_trajectories_for_visualizer = starting_poses.clone()
    inner_data, cost, stacked_trajectories_for_visualizer = sequential((inner_data, cost, stacked_trajectories_for_visualizer))


    # optimizer = optim.Adam(sequential.parameters(), lr=0.0001)
    
    #### OPTIMIZATION ####
    # optimizer = optim.Adam([inner_data], lr=lr, weight_decay =lr/(epochs*0.5)  )
    # TODO: insert inner_data into module
    for optim_number in range(0,10000):
        
        if rospy.is_shutdown():
            break

        inner_data = torch.nn.Parameter(starting_poses.clone())
        stacked_trajectories_for_visualizer = starting_poses.clone()
        inner_data = starting_poses.clone()

        if inner_data.grad is not None:
           inner_data.grad.data.zero_()
        if observed_state.grad is not None:
           observed_state.grad.data.zero_()

        ### FORWARD PASS #### 
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
        # print ("type ", type(inner_data))
        inner_data, cost, stacked_trajectories_for_visualizer = sequential((inner_data, cost, stacked_trajectories_for_visualizer))
            
        #### VISUALIZE ####
        pedestrians_visualizer.publish(inner_data2[1:])
        robot_visualizer.publish(inner_data2[0:1])
        learning_vis.publish(stacked_trajectories_for_visualizer)
        
        if (optim_number % 1 == 0):
            print ("       ---iter # ",optim_number, "  cost", cost.sum().item())

        # optimizer.zero_grad()

        #### CALC GRAD #### 
        cost = cost.sum()
        cost.backward()

        
        #print ("starting_poses: ", starting_poses)
        gradient =  inner_data.grad
        gradient[0,:] *= 0
        print (sequential[0].init_state.grad)
