#!/usr/bin/python3
import rospy
import random
from Visualizer import Visualizer
from Visualizer2 import Visualizer2
import numpy as np

import torch
from forward import calc_forces, pose_propagation, calc_cost_function
from Param import Param
from optimization import  get_poses_probability
from optimisation import Linear
import torch.nn as nn
import time
from utils import check_poses_not_the_same


def optimisation(EPOCHS, lr, starting_poses, goals, robot_init_pose, param):
    print (starting_poses.requires_grad)
    for epoch_numb in range(0, EPOCHS):
        start = time.time()
        if rospy.is_shutdown():
            break

        stacked_trajectories_for_visualizer = starting_poses.clone().detach()
        inner_data = starting_poses.clone().detach()
        inner_data.requires_grad_(True)

        ### FORWARD PASS ####
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)

        # from tensorboardX import SummaryWriter
        # writer = SummaryWriter()
        # writer.add_graph(sequential, ((inner_data, cost, stacked_trajectories_for_visualizer, probability_matrix),))

        # exit()
        
        probability_matrix = get_poses_probability(
            inner_data, param.input_distrib)
        
        _, cost, stacked_trajectories_for_visualizer,_,_,_ = sequential(
            (inner_data, cost, stacked_trajectories_for_visualizer, goals, param, robot_init_pose))

        #### VISUALIZE ####
        if param.do_visualization:
            vis_peds.publish(starting_poses[1:])
            vis_goals_ped.publish(goals)
            vis_robot.publish(starting_poses[0:1])
            vis_learning.publish(stacked_trajectories_for_visualizer)

        #### CALC GRAD ####

        # prob_cost
        # print ("prob_cost ", prob_cost)

        # print ("probability_matrix.grad ", probability_matrix.grad)

        # #print ("starting_poses: ", starting_poses)

        # print ("param.input_state.grad ", param.input_state.grad)

        prob_cost = cost * probability_matrix

        prob_cost.sum().backward()
        gradient = inner_data.grad
        gradient[0, :] *= 0

        # print (starting_poses[1,2:4])
        if gradient is not None:
            with torch.no_grad():
                delta_pose = lr * gradient[1:, 0:2]
                delta_vel = lr * gradient[1:, 2:4]
                delta_pose = torch.clamp(delta_pose, max=0.02, min=-0.02)
                delta_vel = torch.clamp(delta_vel, max=0.02, min=-0.02)
                starting_poses[1:, 0:2] = starting_poses[1:, 0:2] + delta_pose
                starting_poses[1:, 2:4] = starting_poses[1:, 2:4] + delta_vel
        with torch.no_grad():
            for i in range(starting_poses.shape[0]):
                for j in range(i, starting_poses.shape[0]):
                    # check that they are not in the same place
                    if i != j:
                        starting_poses[i, 0:2], starting_poses[j, 0:2] = check_poses_not_the_same(
                            starting_poses[i, 0:2], starting_poses[j, 0:2], gradient[i, 0:2], gradient[j, 0:2], lr)

        if inner_data.grad is not None:
            inner_data.grad.data.zero_()
        if observed_state.grad is not None:
            observed_state.grad.data.zero_()
        if starting_poses.grad is not None:
            starting_poses.grad.data.zero_()
        if (epoch_numb % 10 == 0):
            # print ("probability_matrix:", probability_matrix)
            # print("")
            # print ("cost:", cost)
            # print("")
            print('       ---iter # ', epoch_numb, "      cost: {:.1f}".format(
                prob_cost.sum().item()), '      iter time: ', "{:.3f}".format(time.time()-start))
        del cost
    with torch.no_grad():
        # new_state_propagation=inner_data.detach().clone()
        new_state_propagation = stacked_trajectories_for_visualizer[
            inner_data.shape[0]:inner_data.shape[0]*2].detach().clone()
        new_state_propagation.requires_grad_(True)

        # new_state_propagation[1:,0:2]= new_state_propagation[1:,0:2] + delta_pose
    # new_state_propagation= new_state_propagation[1:,2:4] + delta_vel
    return new_state_propagation, prob_cost


if __name__ == '__main__':
    node = rospy.init_node('mpdm')
    param = Param()
    # torch.autograd.set_detect_anomaly(True)
    # visualisation staff
    vis_peds = Visualizer2('/peds')
    vis_robot = Visualizer2('/robot', color=1)
    vis_goals_ped = Visualizer2(
        '/goals/ped',               size=[0.1, 0.1, 0.5])
    vis_goals_rob = Visualizer2(
        '/goals/rob',    color=1,   size=[0.1, 0.1, 0.5])
    vis_learning = Visualizer2(
        "/peds/learning", color=2,   size=[0.2, 0.2, 1.0], with_text=False)
    # visualisation staff

    # starting env
    observed_state = param.input_state.clone().detach()
    goals = param.goal
    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    robot_init_pose = observed_state[0, 0:2]
    starting_poses = observed_state.clone()
    # starting env

    # network struct
    modules = []
    for i in range(0, 10):
        modules.append(Linear())
    sequential = nn.Sequential(*modules)
    # network struct

    # robot_pose = param.robot_init_pose
    # robot_goal = param.robot_goal

    # #### OPTIMIZATION ####
    global_start_time = time.time()
    robot_init_speed = param.robot_speed
    while not rospy.is_shutdown():
        rospy.sleep(param.loop_sleep)

        # policys = []
        # param.robot_speed = robot_init_speed
        # robot_policys = generate_robot_policys(robot_init_speed)  # TODO: realize that
        # for robot_speed_sample in robot_policys:
        #     param.robot_speed = robot_speed_sample
        #     starting_poses, cost = optimisation(
        #         EPOCHS=5, lr=10**-4, starting_poses=starting_poses, goals=goals, robot_init_pose=robot_init_pose_sample, param=param)
        #     policys.append([param.robot_speed, starting_poses, cost])       

        #### find the best policy
        # best_policy = policys[0]
        # for policy in policys:
        #     if best_policy[2]<policy[1]:
        #         best_policy = policy


        starting_poses, cost = optimisation(
            EPOCHS=5, lr=10**-4, starting_poses=starting_poses, goals=goals, robot_init_pose=robot_init_pose, param=param)
        robot_init_pose = starting_poses[0, 0:2]
        goals = param.generate_new_goal(goals, starting_poses)
        print ("?")

