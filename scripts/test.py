#! /home/robot/miniconda3/envs/bonsai/bin/python

import torch
from force_attract_with_dest import force_goal, pose_propagation, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, calc_forces
from Visualizer2 import Visualizer2
from Param import Param
import rospy
import time
from utils import check_poses_not_the_same
import torch.nn as nn
import torch.optim as optim
import numpy as np


lr = 10**-4
torch.manual_seed(2)

class Linear(nn.Module):
    def __init__(self):                     #input_features, output_features, bias=True):
        super(Linear, self).__init__()
        pass

    def forward(self, input):
        state, cost, stacked_trajectories_for_visualizer = input
        state.detach_()
        rf, af = calc_forces(state, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
        F = rf + af
        out = pose_propagation(F, state, param.DT, param.pedestrians_speed)
        temp = calc_cost_function(param.a, param.b, param.e, goals, robot_init_pose, out, observed_state)
        new_cost = cost + temp.view(-1,1)
        stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer,state.clone()))

        return (out, new_cost, stacked_trajectories_for_visualizer) 


if __name__ == '__main__':

    param = Param()    
    rospy.init_node("vis")
    if param.do_visualization:
        
        pedestrians_visualizer = Visualizer2("peds")
        ped_goals_visualizer = Visualizer2("goals/ped",size=[0.1, 0.1, 0.5])
        robot_visualizer = Visualizer2("robot", color=1)
        learning_vis = Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2, with_text = False)

    
    # torch.autograd.set_detect_anomaly(True)
    observed_state = param.input_state
    goals = param.goal
    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    robot_init_pose = observed_state[0,0:2]#param.robot_init_pose.requires_grad_(True)

    gradient = None

    starting_poses = observed_state.clone()

    ##### MODEL CREATING ######
    modules = []
    for i in range (0, 20):
        modules.append(Linear())

    sequential = nn.Sequential(*modules)


    #### OPTIMIZATION ####
    global_start_time = time.time()
    optim_number = 0
    for optim_number in range(0,200):
        start = time.time()    
        if rospy.is_shutdown():
            break

        stacked_trajectories_for_visualizer = starting_poses.clone()
        inner_data = starting_poses.clone()

        
        ### FORWARD PASS #### 
        
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)

        _, cost, stacked_trajectories_for_visualizer = sequential((inner_data, cost, stacked_trajectories_for_visualizer))
        
        #### VISUALIZE ####
        if param.do_visualization:
            ped_goals_visualizer.publish(goals)
            pedestrians_visualizer.publish(starting_poses[1:])
            robot_visualizer.publish(starting_poses[0:1])
            learning_vis.publish(stacked_trajectories_for_visualizer)
        
        

        # optimizer.zero_grad()

        #### CALC GRAD #### 
        cost = cost.sum()
        cost.backward()
        
        #print ("starting_poses: ", starting_poses)
        gradient =  inner_data.grad
        gradient[0,:] *= 0


        if gradient is not None:
            with torch.no_grad():
                starting_poses[1:,0:2] = starting_poses[1:,0:2] +  torch.clamp(lr *gradient[1:,0:2],max=0.02,min=-0.02)
                starting_poses[1:,2:4] = starting_poses[1:,2:4] +  torch.clamp(10*lr * gradient[1:,2:4],max=0.02,min=-0.02)
                starting_poses.retain_grad()

        for i in range( starting_poses.shape[0]):
            for j in range(i,starting_poses.shape[0]):
                # check that they are not in the same place
                if i != j:
                    starting_poses[i,0:2], starting_poses[j,0:2] = check_poses_not_the_same(starting_poses[i,0:2], starting_poses[j,0:2], gradient[i,0:2], gradient[j,0:2], lr)
        
        if inner_data.grad is not None:
           inner_data.grad.data.zero_()
        if observed_state.grad is not None:
           observed_state.grad.data.zero_()
        if starting_poses.grad is not None:
           starting_poses.grad.data.zero_()
        if (optim_number % 10 == 0):
            print ('       ---iter # ',optim_number, "      cost: {:.1f}".format(cost.sum().item()) ,'      iter time: ', "{:.3f}".format(time.time()-start) )


    print ("delta poses:", observed_state[:,0:2] - starting_poses[:,0:2])
    print ("average time for step: ", (time.time()-global_start_time)/optim_number)