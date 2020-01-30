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

import logging

lr = 10
torch.manual_seed(7)

def come_to_me(origin, point,koef):
    ox, oy = origin
    px, py = point

    qx = px+(ox-px)*koef
    qy = py+(oy-py)*koef
    return qx, qy



def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_poses_probability(agents_pose, agents_pose_distrib, index_X=0 ,index_Y=1) :

    probability = torch.exp(agents_pose_distrib.log_prob(agents_pose))* torch.sqrt(2 * math.pi * agents_pose_distrib.stddev**2)
    probability_ = 0.5*(probability[:,index_X] + probability[:,index_Y])
    probability_ = probability_.view(-1,1).requires_grad_(True)
    return probability_

class Linear(nn.Module):
    def __init__(self):                     #input_features, output_features, bias=True):
        super(Linear, self).__init__()
        pass

    def forward(self, input):

        # input_state, cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose = input
        state, cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose, policy = input
        # state = 1 * input_state       
        rf, af = calc_forces(state, goals, param.pedestrians_speed, param.robot_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
        F = rf + af
        out = pose_propagation(F, state.clone(), param.DT, param.pedestrians_speed, param.robot_speed)
        temp = calc_cost_function(param.a, param.b, param.e, param.goal, robot_init_pose, out, policy)
        new_cost = cost + ( temp.view(-1,1))
        stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer, state.clone()))
        
        return (out, new_cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose, policy) 



# def optimize(epochs, model, starting_poses, robot_init_pose, param, goals, lr, ped_goals_visualizer, initial_pedestrians_visualizer, pedestrians_visualizer, robot_visualizer, learning_vis, initial_ped_goals_visualizer):
def optimize(epochs, model, starting_poses, robot_init_pose, param, goals, lr, ped_goals_visualizer, initial_pedestrians_visualizer, pedestrians_visualizer, robot_visualizer, learning_vis, initial_ped_goals_visualizer, policy= None, do_print = False, device=None):
    for epoch_numb in range(0,epochs):
        if rospy.is_shutdown():
                break


        max_cost = -math.inf
        start = time.time()    
        stacked_trajectories_for_visualizer = starting_poses.clone()
        inner_data = starting_poses.clone().detach()
        inner_data.requires_grad_(True)

        ### FORWARD PASS #### 
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)

        # gpu staff
        cost = cost.to(device)
        # gpu staff
    

        probability_matrix = get_poses_probability(inner_data, param.input_distrib)
        goal_prob = get_poses_probability(goals, param.goal_distrib)
        vel_prob = get_poses_probability(inner_data, param.input_distrib, index_X=2, index_Y=3)
        goal_prob[0] = 1.
        _, cost, stacked_trajectories_for_visualizer, _,_ ,_ ,_= model((inner_data, cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose, policy))
        
        # print (goals)
        #### VISUALIZE ####
        if param.do_visualization and None not in [ped_goals_visualizer, initial_pedestrians_visualizer, pedestrians_visualizer, robot_visualizer, learning_vis, initial_ped_goals_visualizer]:
            ped_goals_visualizer.publish(goals)
            # initial_pedestrians_visualizer.publish(observed_state)
            pedestrians_visualizer.publish(starting_poses[1:])
            robot_visualizer.publish(starting_poses[0:1])
            learning_vis.publish(stacked_trajectories_for_visualizer)
            initial_ped_goals_visualizer.publish(param.goal)

        #### CALC GRAD ####         

        prob_cost  = cost * (probability_matrix) * (goal_prob) * vel_prob

        prob_cost.sum().backward()
        total_cost = prob_cost.sum().item()
        if total_cost > max_cost:
            max_cost = total_cost
        gradient = inner_data.grad
        gradient[0,:] *= 0
        
        if gradient is not None:
            with torch.no_grad():

                delta_pose =  lr * gradient[1:,0:2]
                delta_vel = 100*lr * gradient[1:,2:4]
                delta_pose = torch.clamp(delta_pose,max=0.01,min=-0.01)
                delta_vel  = torch.clamp(delta_vel,max=0.02,min=-0.02)
                starting_poses[1:,0:2] = starting_poses[1:,0:2] + delta_pose
                starting_poses[1:,2:4] = starting_poses[1:,2:4] + delta_vel
                goals.grad[0,:] = goals.grad[0,:] * 0
                
                goals = (goals + torch.clamp(lr *10* goals.grad, max=0.2, min=-0.2))#.requires_grad_(True)

        goals.requires_grad_(True)
 
        if goals.grad is not None:
            goals.grad.data.zero_()
        if inner_data.grad is not None:
            inner_data.grad.data.zero_()

        if starting_poses.grad is not None:
            starting_poses.grad.data.zero_()
        
        if (epoch_numb % 1 == 0):
            if do_print:
                print ('       ---iter # ',epoch_numb, "      cost: {:.2f}".format(prob_cost.sum().item()) ,'      iter time: ', "{:.3f}".format(time.time()-start) )
            if param.do_logging:
                logger.debug('       ---iter # '+ str(epoch_numb) + "      cost: {:.2f}".format(prob_cost.sum().item()) +'      iter time: '+ "{:.3f}".format(time.time()-start))
    return max_cost



if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    param = Param()    
    
    ##### logging init   #####
    if param.do_logging:        
        
        logger = logging.getLogger('optimization.py')
        logger.setLevel(logging.DEBUG)
        
        fh = logging.FileHandler('logs/log.log')
        fh.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info("----------------------------------------------------------")
        logger.info("sim params: num_ped" + str(param.num_ped) + "   num of layers " + str(param.number_of_layers))
        ####### logging init end ######
    
    
    
    rospy.init_node("vis")
    if param.do_visualization:

        pedestrians_visualizer = Visualizer2("peds", starting_id=1)
        initial_pedestrians_visualizer = Visualizer2("peds_initial", color=3, size=[0.6/3, 0.6/3, 1.8/3], with_text = False)
        ped_goals_visualizer = Visualizer2("goals/ped",size=[0.1, 0.1, 0.5])
        initial_ped_goals_visualizer = Visualizer2("goals/init_ped", size=[0.05, 0.05, 0.25], color=3, with_text = True)
        robot_visualizer = Visualizer2("robot", color=1)
        learning_vis = Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2, with_text = False)
    
    else :
        pedestrians_visualizer = None
        initial_pedestrians_visualizer = None
        ped_goals_visualizer = None
        initial_ped_goals_visualizer = None
        robot_visualizer = None
        learning_vis = None



    observed_state = param.input_state.clone().detach()
    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    robot_init_pose = observed_state[0,0:2]#param.robot_init_pose.requires_grad_(True)
    
    goals = param.goal.clone().detach().requires_grad_(True)
    # ##### MODEL CREATING ######
    modules = []
    for i in range (0, param.number_of_layers):
        modules.append(Linear())

    sequential = nn.Sequential(*modules)
    # #### OPTIMIZATION ####
    global_start_time = time.time()

    starting_poses = observed_state.clone()
    param.optim_epochs = 10000
    cost = optimize(param.optim_epochs, sequential ,starting_poses, robot_init_pose, param,goals ,lr,ped_goals_visualizer, initial_pedestrians_visualizer, pedestrians_visualizer, robot_visualizer, learning_vis, initial_ped_goals_visualizer, do_print=True,)
    
    print ("average time for step: ", (time.time()-global_start_time)/(param.optim_epochs+1))

# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
# writer.add_graph(sequential, ((inner_data, cost, stacked_trajectories_for_visualizer, probability_matrix),))
# exit()