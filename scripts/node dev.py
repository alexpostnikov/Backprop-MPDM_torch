#!/usr/bin/python3
import rospy
import torch
from Utils.RvizSub import RvizSub
from Utils.Visualizer2 import Visualizer2
from MPDM.SFM import SFM
from MPDM.RepulsiveForces import RepulsiveForces
from MPDM.Policy import Policy
from Param import Param
from optimization import Linear
import torch.nn as nn
import time
from multiprocessing import Process
lr = 10

if __name__ == '__main__':
    rospy.init_node("mpdm")
    # MPDM core
    param = Param(device)    
    rep_f = RepulsiveForces()
    sfm = SFM(rep_f)
    policy_calc = Policy()
    # MPDM core

    if param.do_visualization:

        pedestrians_visualizer = Visualizer2("peds", starting_id=1)
        initial_pedestrians_visualizer = Visualizer2("peds_initial", color=3, size=[
                                                     0.6/3, 0.6/3, 1.8/3], with_text=False)
        ped_goals_visualizer = Visualizer2("goals/ped", size=[0.1, 0.1, 0.5])
        initial_ped_goals_visualizer = Visualizer2(
            "goals/init_ped", size=[0.05, 0.05, 0.25], color=3, with_text=True)
        robot_visualizer = Visualizer2("robot", color=1, with_text=False)
        policy_visualizer = Visualizer2(
            "robot_policy", color=1,  with_text=False)
        learning_vis = Visualizer2(
            "peds/learning", size=[0.2, 0.2, 1.0], color=2, with_text=False)

    modules = []
    for i in range(0, param.number_of_layers):
        modules.append(Linear())

    manual_goal = [None, None]
    manual_pose = [None, None]
    goal_sub = RvizSub(manual_goal, manual_pose)

    sequential = nn.Sequential(*modules)
    # gpu staff
    sequential = sequential.to(device)
    # gpu staff
    for i in range (1000):
        if rospy.is_shutdown():
            break
        observed_state = param.input_state.clone().detach()
        # observed_state = observed_state.to(device)
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
        # param.robot_init_pose.requires_grad_(True)
        robot_init_pose = observed_state[0, 0:2]
        goals = param.goal.clone().detach().requires_grad_(False)

        # ##### MODEL CREATING ######
        modules = []
        for i in range(0, param.number_of_layers):
            modules.append(Linear())

        sequential = nn.Sequential(*modules)
        # gpu staff
        sequential = sequential.to(device)
        param.to_device(device)
        # gpu staff
        # #### OPTIMIZATION ####
        global_start_time = time.time()

        # policies = ["stop", "go-solo", "left", "right"]#, "fast", "slow"]
        policies = ["go-solo"]
        results = []
        processes = []
        start = time.time()
        for policy in policies:
            results.append(policy_calc.get_policy_cost(sequential, observed_state, goals, param, lr, ped_goals_visualizer,
                  initial_pedestrians_visualizer,
                  pedestrians_visualizer, robot_visualizer,
                  learning_vis, initial_ped_goals_visualizer, policy))
        #     processes.append(Process(target=func_for_Pool, args=(sequential, observed_state, goals, param, lr, ped_goals_visualizer,
        #           initial_pedestrians_visualizer,
        #           pedestrians_visualizer, robot_visualizer,
        #           learning_vis, initial_ped_goals_visualizer,)))
        #     processes[-1].start()            

        # for process in processes:
        #     process.join()

            
            '''
            starting_poses = observed_state.clone()
            # gpu staff
            starting_poses = starting_poses.to(device)
            # gpu staff

            starting_poses, goals_in_policy, param.robot_speed = apply_policy(policy, starting_poses, goals.clone(), param.robot_speed)
            goals_in_policy.requires_grad_(True)
            # gpu staff
            goals_in_policy = goals_in_policy.to(device)
            starting_poses = starting_poses.to(device)
            # gpu staff

            cost = optimize(param.optim_epochs, sequential, starting_poses, 
                param.robot_init_pose, param, 
                goals_in_policy ,lr,ped_goals_visualizer, 
                initial_pedestrians_visualizer, 
                pedestrians_visualizer, robot_visualizer, 
                learning_vis, initial_ped_goals_visualizer, policy, device = device)
                
            results.append(cost)
            '''
        # print(np.array(results) - min(results))
        pol_numb = results.index(min(results))
        policy = policies[pol_numb]
        print(policy)
        print ("one step of choosing policy took ", "{:.1f}".format(- start + time.time()), "sec")
        
        with torch.no_grad():
            param.input_state, goals, param.robot_speed = policy_calc.apply_policy(
                policy, param.input_state, goals.clone(), param.robot_speed)

        with torch.no_grad():
            # propagate
            rf, af = sfm.calc_forces(param.input_state, goals, param.pedestrians_speed, param.robot_speed,
                                 param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
            F = rf + af
            observed_state_new = sfm.pose_propagation(
                F, param.input_state, param.DT, param.pedestrians_speed, param.robot_speed)

            # update scene with new poses & check is goals achieved
            param.update_scene(observed_state_new, param.goal)
            goals = param.goal.clone()
            goals = param.generate_new_goal(goals, param.input_state)

            if manual_goal[0] is not None:
                param.goal[0, 0] = manual_goal[0]
                param.goal[0, 1] = manual_goal[1]
                manual_goal[0] = None

            if manual_pose[0] is not None:
                param.input_state[0, 0] = manual_pose[0]
                param.input_state[0, 1] = manual_pose[1]

                # param.robot_init_pose[0] = manual_pose[0]
                # param.robot_init_pose[1] = manual_pose[1]
                manual_pose[0] = None

            param.robot_init_pose = param.input_state[0, 0:2]

            goals.requires_grad_(True)
            # print("goals",goals)
        policy_visualizer.publish(param.input_state[0:1],  policy)
