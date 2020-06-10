from MPDM.Layer import Linear
import torch.nn as nn
import torch
import numpy as np
import math
# TODO: that ROS part woldn`t need to be here
from Utils.Visualizer3 import Visualizer3
from Utils.Visualizer4 import Visualizer4


class MPDM:
    def __init__(self, param, transition_model, covariance_model=None, policys=None):
        self.param = param
        self.map = None
        self.policys = policys
        self.modules = []
        self.path = None
        self.goals = None
        self.states = None
        self.prev_states = None
        ###### MODEL CREATING ######
        for _ in range(self.param.number_of_layers):
            self.modules.append(Linear(transition_model, covariance_model))
        self.sequential = nn.Sequential(*self.modules)

    def is_init(self):
        return self.states is not None

    def update_state(self, robot: np.ndarray, peds: np.ndarray, robot_goal: np.ndarray, peds_goals: np.ndarray, map=None):
        try:
            self.prev_states = self.states.copy()
        except:
            pass
        try:
            self.map = map
            states = [robot]
            goals = [robot_goal]
            if peds is not None and len(peds) > 1:
                for i in range(len(peds)):
                    states.append(peds[i])
                    goals.append(peds_goals[i])
            self.goals = torch.Tensor(goals)
            self.states = torch.Tensor(states)
        except:
            self.states = None
        # [
        # robot [x,y,yaw,vx,vy,vyaw],
        # ped1 [x,y,yaw,vx,vy,vyaw],
        # ped2 [x,y,yaw,vx,vy,vyaw],
        # ...
        # ]

    def predict(self, epoch=10):
        cost, array_path = self.optimize(epoch)
        self.path = array_path
        return self.path

    def get_robot_path(self,):
        return self.path

    def optimize(self, epochs):
        if self.states is None:
            print("\t Warn: states is None!")
            return None

        max_cost = -math.inf
        max_cost_path = None
        starting_poses = self.states.clone().detach()
        for epoch_numb in range(0, epochs):
            inner_data = starting_poses.clone().detach()
            inner_data.requires_grad_(True)
            goals = self.goals.clone().detach()
            goals.requires_grad_(True)
            ### FORWARD PASS ####
            # cost = torch.zeros(len(inner_data)-1, 1).requires_grad_(True)
            cost = torch.zeros(len(inner_data-1), 1).requires_grad_(True)
            probability_matrix, goal_prob, vel_prob = self.get_probability(
                inner_data, goals, self.param)
            goal_prob[0] = 1.  # robot goal probability
            stacked_covariance = np.zeros((1, len(inner_data), 2)).tolist()
            stacked_state = [inner_data.clone()]
            # TODO: stopped here
            inner_data_, stacked_state, cost, stacked_covariance, _, _ = self.sequential(
                (inner_data, stacked_state, cost, stacked_covariance, goals, self.policys))

            # _, cost, stacked_covariance, stacked_trajectories, stacked_trajectories_vis, _, _, _ = self.sequential(
            #     (inner_data, cost, stacked_covariance, stacked_trajectories, stacked_trajectories_vis, goals, robot_init_pose, self.policys))

            # print (goals)
            #### VISUALIZE ####
            # stidno, stidno....
            # stacked_covariance_vis = []
            # for x in stacked_covariance:
            #     for y in x:
            #         stacked_covariance_vis.append(y)
            # stacked_covariance_vis[:,1]= stacked_covariance_vi
            # stidno, stidno....
            # TODO: that ROS part woldn`t need to be here
            # if self.visualize:
            #     self.covariance_vis.publish(
            #         stacked_trajectories_vis.clone().detach(), stacked_covariance_vis)
            #     self.ped_goals_visualizer.publish(goals.clone().detach())
            #     # initial_pedestrians_visualizer.publish(observed_state)
            #     # self.pedestrians_visualizer.publish(
            #     #     inner_data[1:].clone().detach())
            #     self.robot_visualizer.publish(inner_data[0:1].clone().detach())
            #     self.learning_vis.publish(
            #         stacked_trajectories_vis.clone().detach())
            #     self.initial_ped_goals_visualizer.publish(
            #         self.goals.clone().detach())

            # TODO: that ROS part woldn`t need to be here

            #### CALC GRAD ####

            prob_cost = cost * (probability_matrix) * (goal_prob) * vel_prob
            # torch.autograd.set_detect_anomaly(True)
            prob_cost.sum().backward()
            total_cost = prob_cost.sum().item()
            if total_cost > max_cost:
                max_cost = total_cost
                # max_cost_path = stacked_trajectories_vis.clone().detach(
                # )[::len(inner_data)]  # <== only robot positions+forces
            gradient = inner_data.grad
            print ("gradient = ", gradient)
            if gradient is not None:
                gradient[0, :] *= 0
                with torch.no_grad():

                    delta_pose = self.param.lr * gradient[1:, 0:2]
                    delta_vel = 100*self.param.lr * gradient[1:, 2:4]
                    delta_pose = torch.clamp(delta_pose, max=0.01, min=-0.01)
                    delta_vel = torch.clamp(delta_vel, max=0.02, min=-0.02)
                    starting_poses[1:, 0:2] = starting_poses[1:,
                                                             0:2] + delta_pose
                    starting_poses[1:, 2:4] = starting_poses[1:,
                                                             2:4] + delta_vel
                    goals.grad[0, :] = goals.grad[0, :] * 0

                    goals = (goals + torch.clamp(self.param.lr * 10 * goals.grad,
                                                 max=0.2, min=-0.2))  # .requires_grad_(True)

            goals.requires_grad_(True)

            if goals.grad is not None:
                goals.grad.data.zero_()
            if inner_data.grad is not None:
                inner_data.grad.data.zero_()
            # if starting_poses.grad is not None:
            #     starting_poses.grad.data.zero_()

        return max_cost, max_cost_path

    def get_probability(self, inner_data, goals, param):

        # pose
        num_ped = len(inner_data)  # -1
        data_shape = len(inner_data[0])
        vel_shape = int(data_shape/2)
        pose_shape = vel_shape

        input_state_std = param.pose_std_coef * \
            torch.rand((num_ped, data_shape))
        input_state_std[:, pose_shape:(pose_shape+vel_shape)] = param.velocity_std_coef * \
            torch.rand((num_ped, vel_shape))
        agents_pose_distrib = torch.distributions.normal.Normal(
            inner_data, input_state_std)

        # position
        index_X, index_Y, index_YAW = 0, 1, 2
        probability = torch.exp(agents_pose_distrib.log_prob(
            inner_data)) * torch.sqrt(2 * math.pi * agents_pose_distrib.stddev**2)
        probability_ = 0.5 * \
            (probability[:, index_X] + probability[:,
                                                   index_Y] + probability[:, index_YAW])
        probability_matrix = probability_.view(-1, 1).requires_grad_(True)
        # velocity
        index_VX, index_VY, index_VYAW = 3, 4, 5
        probability_ = 0.5 * \
            (probability[:, index_VX] + probability[:,
                                                    index_VY] + probability[:, index_VYAW])
        vel_prob = probability_.view(-1, 1).requires_grad_(True)
        # goal
        goal_std = param.goal_std_coef * torch.rand((num_ped, pose_shape))
        goal_distrib = torch.distributions.normal.Normal(goals, goal_std)
        index_X, index_Y, index_YAW = 0, 1, 2
        probability = torch.exp(goal_distrib.log_prob(
            goals)) * torch.sqrt(2 * math.pi * goal_distrib.stddev**2)
        probability_ = 0.5 * \
            (probability[:, index_X] + probability[:,
                                                   index_Y] + probability[:, index_YAW])
        goal_prob = probability_.view(-1, 1).requires_grad_(True)

        return probability_matrix, goal_prob, vel_prob
