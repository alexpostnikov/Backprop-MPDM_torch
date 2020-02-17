#! /home/robot/miniconda3/envs/bonsai/bin/python

import torch
import math
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
torch.manual_seed(9)

def get_poses_probability(agents_pose, agents_pose_distrib):
    
    # print ("agents_pose ", agents_pose[9])
    probability = torch.exp(agents_pose_distrib.log_prob(agents_pose))* torch.sqrt(2 * math.pi * agents_pose_distrib.stddev**2)
    # print ("probability ", probability[3])
    probability_ = (probability[:,0:1] * probability[:,1:2]).requires_grad_(True)
    
    # probability__ = torch.clamp(probability_, max=1.0, min=0.0).requires_grad_(True)
    # print ("probability__ ", probability__)
    return probability_

class Linear(nn.Module):
    def __init__(self):                     #input_features, output_features, bias=True):
        super(Linear, self).__init__()
        pass

    def forward(self, input):

        state, cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose = input
        rf, af = calc_forces(state, goals, param.pedestrians_speed, param.robot_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)

        F = rf + af
        # if probability_matrix is None:

        out = pose_propagation(F, state, param.DT, param.pedestrians_speed)

        temp = calc_cost_function(param.a, param.b, param.e, goals, robot_init_pose, out)
        #probability_matrix.view(-1,1)
        new_cost = cost + ( temp.view(-1,1))

        stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer,state.clone()))

        return (out, new_cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose) 

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    param = Param()    
    goals = param.goal.requires_grad_(True)
    rospy.init_node("vis")
    if param.do_visualization:

        pedestrians_visualizer = Visualizer2("peds", starting_id=1)
        initial_pedestrians_visualizer = Visualizer2("peds_initial", color=3, size=[0.6/3, 0.6/3, 1.8/3], with_text = False)
        ped_goals_visualizer = Visualizer2("goals/ped",size=[0.1, 0.1, 0.5])
        initial_ped_goals_visualizer = Visualizer2("goals/init_ped", size=[0.05, 0.05, 0.25], color=3, with_text = True)
        robot_visualizer = Visualizer2("robot", color=1)
        learning_vis = Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2, with_text = False)


    observed_state = param.input_state.clone().detach()


    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    robot_init_pose = observed_state[0,0:2]#param.robot_init_pose.requires_grad_(True)

    # gradient = None

    starting_poses = observed_state.clone()

    # ##### MODEL CREATING ######
    modules = []
    for i in range (0, 10):
        modules.append(Linear())

    sequential = nn.Sequential(*modules)


    # #### OPTIMIZATION ####
    global_start_time = time.time()
    # epoch_numb = 0
    for epoch_numb in range(0,1000):
        start = time.time()    
        if rospy.is_shutdown():
            break

        stacked_trajectories_for_visualizer = starting_poses.clone()
        inner_data = starting_poses.clone().detach()
        inner_data.requires_grad_(True)


        ### FORWARD PASS #### 
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)

        # from tensorboardX import SummaryWriter
        # writer = SummaryWriter()
        # writer.add_graph(sequential, ((inner_data, cost, stacked_trajectories_for_visualizer, probability_matrix),))
        # exit()
        probability_matrix = get_poses_probability(inner_data, param.input_distrib)
        goal_prob = get_poses_probability(goals, param.goal_distrib)
        _, cost, stacked_trajectories_for_visualizer = sequential((inner_data, cost, stacked_trajectories_for_visualizer))



        #### VISUALIZE ####
        if param.do_visualization:
            ped_goals_visualizer.publish(goals)
            initial_pedestrians_visualizer.publish(observed_state)
            pedestrians_visualizer.publish(starting_poses[1:])
            robot_visualizer.publish(starting_poses[0:1])
            learning_vis.publish(stacked_trajectories_for_visualizer)
            initial_ped_goals_visualizer.publish(param.goal)
        
        
        #### CALC GRAD ####         

        prob_cost  = cost * torch.sqrt(probability_matrix) * torch.sqrt(goal_prob)
        print (goal_prob[4])
        prob_cost.sum().backward()
        gradient = inner_data.grad
        gradient[0,:] *= 0
        
        if gradient is not None:
            with torch.no_grad():
                delta_pose = lr *gradient[1:,0:2]

                delta_vel = lr * gradient[1:,2:4]
                delta_pose = torch.clamp(delta_pose,max=0.01,min=-0.01)
                delta_vel = torch.clamp(delta_vel,max=0.02,min=-0.02)
                starting_poses[1:,0:2] = starting_poses[1:,0:2] + delta_pose
                
                goals = (goals + torch.clamp(lr * goals.grad,max=0.2,min=-0.2)).requires_grad_(True)
        
        if inner_data.grad is not None:
           inner_data.grad.data.zero_()
        if observed_state.grad is not None:
           observed_state.grad.data.zero_()
        if starting_poses.grad is not None:
           starting_poses.grad.data.zero_()
        if (epoch_numb % 1 == 0):

            print ('       ---iter # ',epoch_numb, "      cost: {:.1f}".format(prob_cost.sum().item()) ,'      iter time: ', "{:.3f}".format(time.time()-start) )



    print ("delta poses:", observed_state[:,0:2] - starting_poses[:,0:2])
    print ("average time for step: ", (time.time()-global_start_time)/epoch_numb)