from MPDM.Layer import Linear
import torch.nn as nn
import torch
import numpy as np
import math
import time



class MPDM:
    def __init__(self, param, transition_model, covariance_model=None, policies=None):
        self.param = param
        self.map = None
        self.policies = policies
        self.modules = []
        self.path = None
        self.goals = None
        self.states = None
        self.prev_states = None
        self.learning_stacked_state = []
        self.learning_stacked_cost = []
        self.learning_stacked_covariance = []
        self.learning_stacked_policys = []
        self.prev_goals = None
        ###### MODEL CREATING ######
        for _ in range(self.param.number_of_layers):
            self.modules.append(Linear(transition_model, covariance_model))
        self.sequential = nn.Sequential(*self.modules)

    def is_init(self):
        return self.states is not None

    def update_state(self, robot: np.ndarray, peds: np.ndarray, robot_goal: np.ndarray, peds_goals: np.ndarray, map=None):

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
        self.optimize(epoch)
        best_epoch = self.learning_stacked_cost.index(
            min(self.learning_stacked_cost))
        self.path = torch.stack(self.learning_stacked_state)[
            best_epoch, :, 0]  # suk
        return self.path

    def get_robot_path(self,):
        return self.path

    def optimize(self, epochs):
        # torch.autograd.set_detect_anomaly(True)
        if self.states is None:
            print("\t Warn: states is None!")
            return None

        max_cost = -math.inf
        max_cost_path = None

        self.learning_stacked_state = []
        self.learning_stacked_cost = []
        self.learning_stacked_covariance = []
        self.learning_stacked_policys = []
        self.propogation_times = []

        for policy in self.policies:
            sub_states, sub_goals = policy.apply(self.states.clone(), self.goals.clone())  # self.apply_policy(policy)
            self.do_epochs(epochs, sub_goals, sub_states, policy)

    def do_epochs(self, epochs, sub_goals, sub_states, policy):
        starting_poses = sub_states.clone().detach()
        for epoch_numb in range(0, epochs):
            start = time.time()
            inner_data = starting_poses.clone().detach()
            inner_data.requires_grad_(True)
            goals = sub_goals.clone().detach()
            goals.requires_grad_(True)

            ### FORWARD PASS ####
            cost = torch.zeros(len(inner_data - 1), 1).requires_grad_(True)
            probability_matrix, goal_prob, vel_prob = self.get_probability(
                inner_data, goals, self.param)
            goal_prob[0] = 1.  # robot goal probability
            stacked_covariance = np.zeros((1, len(inner_data), 2)).tolist()
            stacked_state = [inner_data.clone()]
            inner_data_, stacked_state, cost, stacked_covariance, _ = self.sequential(
                (inner_data, stacked_state, cost, stacked_covariance, goals))

            self.learning_stacked_policys.append(policy.name)
            self.learning_stacked_state.append(torch.stack(stacked_state))
            # maybe need to append total_cost instead of cost
            self.learning_stacked_covariance.append(stacked_covariance)

            #### CALC GRAD ####

            prob_cost = cost * probability_matrix * goal_prob * vel_prob
            prob_cost.sum().backward()

            self.learning_stacked_cost.append(float(prob_cost.sum()))
            gradient = inner_data.grad
            # print ("gradient = ", gradient)
            if gradient is not None:
                gradient[0, :] *= 0
                with torch.no_grad():

                    delta_pose = self.param.lr * gradient[1:, 0:2]
                    delta_vel = 100 * self.param.lr * gradient[1:, 2:4]
                    delta_pose = torch.clamp(delta_pose, max=0.1, min=-0.1)
                    delta_vel = torch.clamp(delta_vel, max=0.2, min=-0.2)
                    starting_poses[1:, 0:2] = starting_poses[1:,
                                              0:2] + delta_pose
                    starting_poses[1:, 2:4] = starting_poses[1:,
                                              2:4] + delta_vel
                    goals.grad[0, :] = goals.grad[0, :] * 0

                    goals = (goals + torch.clamp(self.param.lr * 10 * goals.grad,
                                                 max=0.2, min=-0.2))  # .requires_grad_(True)

            else:
                print("gradient is None!")
            goals.requires_grad_(True)

            if goals.grad is not None:
                goals.grad.data.zero_()
            if inner_data.grad is not None:
                inner_data.grad.data.zero_()
            self.propogation_times.append(time.time() - start)
        print("{policy} cost = {cost:0.4f}".format(policy=policy.name, cost=torch.sum(cost).item()))

    def get_learning_data(self):
        return self.learning_stacked_state, self.goals, self.learning_stacked_cost, self.learning_stacked_covariance, self.learning_stacked_policys, self.propogation_times

    # TODO: go out to fake_publicator
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


# measure risks from covariances
# acceleration
# different
# recurent net
