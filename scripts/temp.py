#! /home/robot/miniconda3/envs/bonsai/bin/python

import sys
print (sys.version)
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
lr = 5*10**-6
class Linear(nn.Module):
    def __init__(self):#, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        #self.input_features = input_features
        #self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.

        #self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        #if bias:
        #    self.bias = nn.Parameter(torch.Tensor(output_features))
        #else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
        #    self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        #self.weight.data.uniform_(-0.1, 0.1)
        #if bias is not None:
        #    self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        state = input[0]
        cost = input[1]
        stacked_trajectories_for_visualizer = input[2]
        param = input[3]
        rf, af = calc_forces(state, param.goal, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
        F = rf + af
        output = pose_propagation(F, state, param.DT, param.pedestrians_speed)
        temp = calc_cost_function(param.a, param.b, param.e, param.goal, state[0,0:2], output, observed_state)
        cost = cost + temp
        stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer,inner_data.clone()))
        # See the autograd section for explanation of what happens here.
        return (output, cost, stacked_trajectories_for_visualizer, param) #LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    rospy.init_node("vis")
    pedestrians_visualizer = Visualizer2("peds")
    torch.manual_seed(1)
    param = Param()
    look_ahead_steps = int(param.look_ahead_seconds/ param.DT)
    # torch.autograd.set_detect_anomaly(True)
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

    inner_data = starting_poses.clone()
    # loss_fn  = torch.nn.L1Loss()
    
    
    # optimizer = optim.Adam([inner_data], lr=0.0001)
    lr = 0.0001
    epochs = 100
    # optimizer = optim.Adam([inner_data], lr=lr, weight_decay =lr/(epochs*0.5)  )
    # optimizer = optim.SGD([inner_data], lr=0.0001, momentum=0.9)
    # optimizer = optim.SGD([
    #             {'params': [inner_data]}
    #         ], lr=1e-2, momentum=0.9)

    first_time = True
    #### OPTIMIZATION ####
    # optimizer = optim.Adam([inner_data], lr=lr, weight_decay =lr/(epochs*0.5)  )
    # TODO: insert inner_data into module
    for optim_number in range(0,10000):
        
        if rospy.is_shutdown():
            break

        stacked_trajectories_for_visualizer = starting_poses.clone()
        inner_data = starting_poses.clone()

        

        if inner_data.grad is not None:
           inner_data.grad.data.zero_()
        if observed_state.grad is not None:
           observed_state.grad.data.zero_()

        ### FORWARD PASS #### 
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
        inner_data2, cost, stacked_trajectories_for_visualizer, param = sequential((inner_data, cost, stacked_trajectories_for_visualizer, param))
            
        #### VISUALIZE ####
        pedestrians_visualizer.publish(inner_data2[1:])
        robot_visualizer.publish(inner_data2[0:1])
        learning_vis.publish(stacked_trajectories_for_visualizer)
        
        if (optim_number % 1 == 0):
            print ("       ---iter # ",optim_number, "  cost", cost.sum().item())


        #### CALC GRAD #### 
        # print("cost",cost)
        # exit()
        cost = -cost.sum()
        cost.backward()
        optimizer.zero_grad()
        # loss = loss_fn(cost,torch.tensor(np.inf))
        # if first_time:
        #     loss.backward(retain_graph=True)
        #     first_time = False
        # else:
        # loss.backward()
        #print (loss.grad)
        #print ("starting_poses: ", starting_poses)
        # print (inner_data.grad)
        optimizer.step()
        #print ("starting_poses: ", starting_poses)
        #gradient =  inner_data.grad
        #gradient[0,:] *= 0


        #### APPLY GRAD #### 
        #if gradient is not None:
        #    with torch.no_grad():
        #        starting_poses[1:,0:2] = starting_poses[1:,0:2] +  torch.clamp(lr *gradient[1:,0:2],max=0.02,min=-0.02)
        #        starting_poses[1:,2:4] = starting_poses[1:,2:4] +  torch.clamp(1000*lr * gradient[1:,2:4],max=0.02,min=-0.02)
        #        starting_poses.retain_grad()

        ##### CHECK & FIX  COLLIZIONS ####
        #for i in range( starting_poses.shape[0]):
        #    for j in range(i,starting_poses.shape[0]):
                # check that they are not in the same place
        #        if i != j:
        #            starting_poses[i,0:2], starting_poses[j,0:2] = check_poses_not_the_same(starting_poses[i,0:2], starting_poses[j,0:2], gradient[i,0:2], gradient[j,0:2], lr)


    # ####################
    # inner_data = data.clone().detach()
    # inner_data.retain_grad()
    print ("delta poses:", observed_state[:,0:2] - starting_poses[:,0:2])
    # rf, af = calc_forces(inner_data, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
    # F = rf + af
    # inner_data = pose_propagation(F, inner_data, param.DT, param.pedestrians_speed)


    ###### 
    # data = torch.rand((4,4)).requires_grad_(True)
    # data.retain_grad()
    # cost=calc_cost_function(param.a, param.b, param.e, goals, data[0,0:2], data.clone())
    # cost = cost.view(-1,1)
    # cost = cost.sum()
    
    # # cost.backward(data[:,0:1], retain_graph=True)
    # cost.backward()
    # gradient = data.grad
    # print ("gradient:", gradient)
    
    # cost.backward(inner_data[:,0:2])
    # gradient =  inner_data.grad
    # print ("       ---grad:\n", gradient)

    # gradient =  data.grad
    # print ("       ---grad:\n", gradient)
    # inner_data.grad.data.zero_()
    # data.grad.data.zero_()



    # cost.backward(data[:,2:4])
    # print ("grad" , data.grad)
