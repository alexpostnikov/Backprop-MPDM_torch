from MPDM.Layer import Linear
import torch.nn as nn
import torch
import numpy as np
import math
import time
from typing import Tuple, Optional
from Utils.RosMapTools import get_area, get_areas


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
        self.learning_stacked_forces = []
        self.prev_goals = None
        self.prob_calculator = ProbabilityCalculator()
        ###### MODEL CREATING ######
        for _ in range(self.param.number_of_layers):
            self.modules.append(Linear(transition_model, covariance_model))
        self.sequential = nn.Sequential(*self.modules)

    def is_init(self):
        return self.states is not None and self.goals is not None

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
            self.prob_calculator.update_state(torch.Tensor(np.append(robot, peds).reshape(-1, 6)))
            self.prob_calculator.update_goal(torch.Tensor(np.append(robot_goal, peds_goals).reshape(-1, 3)))
            self.prob_calculator.update_std_rand()
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
        self.learning_stacked_forces = []

        for policy in self.policies:
            sub_states, sub_goals = policy.apply(self.states.clone(), self.goals.clone())  # self.apply_policy(policy)
            self.do_epochs(epochs, sub_goals, sub_states, policy)

    def do_epochs(self, epochs, sub_goals, sub_states, policy, additional_solo_cost = -0.1):
        # 

        starting_poses = sub_states.clone().detach()

        for epoch_numb in range(0, epochs):
            start = time.time()
            inner_data = starting_poses.clone().detach()
            inner_data.requires_grad_(True)
            goals = sub_goals.clone().detach()
            goals.requires_grad_(True)

            ### FORWARD PASS ####
            cost = torch.zeros(len(inner_data - 1), 1).requires_grad_(True)
            prob = self.prob_calculator.get_prob(inner_data.clone(), goals.clone())
            prob[0] = 1
            # probability_matrix, goal_prob, vel_prob = self.get_probability(
            #     inner_data, goals, self.param)
            # goal_prob[0] = 1.  # robot goal probability
            stacked_covariance = np.zeros((1, len(inner_data), 2)).tolist()
            stacked_forces = np.zeros((1, 3,len(inner_data),3)).tolist()#[ster,force_type,ped,xyz]
            stacked_state = [inner_data.clone()]
            map_data = None
            if self.map is not None:
                points = starting_poses[:,0:2].tolist() # position of center
                area = 0.7 # area in meters
                maps_data, resolution = get_areas(self.map, points, area)
                map_origin = None # None == take robot position as center of map
            inner_data_, stacked_state, cost, stacked_covariance, stacked_forces, _,_,_,_,_ = self.sequential(
                (inner_data, stacked_state, cost, stacked_covariance, stacked_forces, self.goals, goals, maps_data, resolution, map_origin))

            self.learning_stacked_policys.append(policy.name)
            self.learning_stacked_state.append(torch.stack(stacked_state))
            # maybe need to append total_cost instead of cost
            self.learning_stacked_covariance.append(stacked_covariance)
            self.learning_stacked_forces.append(stacked_forces)


            #### CALC GRAD ####

            prob_cost = cost * prob
            prob_cost.sum().backward()

            self.learning_stacked_cost.append(float(prob_cost.sum()))
            # TODO: Works better with this hack. it needed to understanding
            if "solo" in policy.name:
                self.learning_stacked_cost[-1]+=additional_solo_cost
            # 
            gradient = inner_data.grad
            # print ("gradient = ", gradient)

            if gradient is not None:
                gradient[0, :] *= 0
                with torch.no_grad():

                    delta_pose = self.param.lr * gradient[1:, 0:2]
                    # delta_vel = 100 * self.param.lr * gradient[1:, 2:4]
                    delta_pose = torch.clamp(delta_pose, max=1, min=-1)
                    # delta_vel = torch.clamp(delta_vel, max=2, min=-2)
                    starting_poses[1:, 0:2] = starting_poses[1:,
                                              0:2] + delta_pose
                    # starting_poses[1:, 2:4] = starting_poses[1:,
                    #                           2:4] + delta_vel
                    goals.grad[0, :] = goals.grad[0, :] * 0

                    goals = (goals + torch.clamp(self.param.lr * goals.grad,
                                                 max=2, min=2))  # .requires_grad_(True)

            else:
                print("gradient is None!")
            goals.requires_grad_(True)

            if goals.grad is not None:
                goals.grad.data.zero_()
            if inner_data.grad is not None:
                inner_data.grad.data.zero_()
            self.propogation_times.append(time.time() - start)
        # NOTE: return for policy debug
        # print("{policy} cost = {cost:0.4f}".format(policy=policy.name, cost=torch.sum(cost).item()))

    def get_learning_data(self):
        return self.learning_stacked_state, np.array(self.goals), self.learning_stacked_cost, self.learning_stacked_covariance, self.learning_stacked_policys, self.propogation_times, self.learning_stacked_forces

    # TODO: go out to fake_publicator
    def get_probability(self, inner_data: torch.Tensor, goals: torch.Tensor, param: object) -> Tuple[torch.Tensor]:
        """
        :param inner_data: torch tensor consited of  pose and velocity [x ,y ,thetta, Vx, Vy] for each ped&robot
        :param goals: torch tensor consited of goal pose [x ,y ,thetta] for each ped&robot
        :param param: object of class Param
        :return: 3 tensors, representing probabilities of current poses, current goals and velocities
        """

        # pose
        num_ped = len(inner_data)  # -1
        data_shape = len(inner_data[0])
        vel_shape = 2  # Vx, Vy
        pose_shape = 3  # x, y, thetta

        # generate random state standard deviation for poses
        input_state_std = param.pose_std_coef * \
            torch.rand((num_ped, data_shape))

        # generate random state standard deviation for velocites
        input_state_std[:, pose_shape:(pose_shape+vel_shape)] = param.velocity_std_coef * \
            torch.rand((num_ped, vel_shape))
        # generate Normal distribution from (mean , std)
        agents_pose_distrib = torch.distributions.normal.Normal(
            inner_data, input_state_std)

        # calculate position probability as a vector of shape [num_agents]
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

        # calculate velocity probability as a vector of shape [num_agents]
        vel_prob = probability_.view(-1, 1).requires_grad_(True)

        # calculate goal probability as a vector of shape [num_agents]
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


class ProbabilityCalculator:
    """
        Calculates probabilities of being in scpecified state, given mean and standart deviations
        (assumpted multivariate gaussian distribs)
    """

    def __init__(self, state: Optional[torch.Tensor] = None, pose_std: Optional[torch.Tensor] = None,
                 angular_std: Optional[torch.Tensor] = None, velocity_std: Optional[torch.Tensor] = None,
                 goal:Optional[torch.Tensor] = None, goal_std: Optional[torch.Tensor] = None):
        """
        :param state: torch tensor with state [x, y, thetta, Vx, Vy] for each person
        :param pose_std: :param state: torch tensor with std state. shape 4x2x2
        :param goal:
        :param goal_std:
        """
        self.num_states = 5
        self.state = state
        self.pose_std = pose_std
        self.angular_std = angular_std
        self.velocity_std = velocity_std

        self.goal = goal
        self.goal_std = goal_std

    def update_state(self, state: torch.Tensor):
        self.state = state

    def update_pose_std(self, std: torch.Tensor):
        self.pose_std = std

    def update_angular_std(self, std: torch.Tensor):
        self.angular_std = std

    def update_velocity_std(self, std: torch.Tensor):
        self.velocity_std = std

    def update_std_rand(self):
        """
        Generate random standard deviations
        """
        a = torch.rand(self.state.shape[0], 2)
        b = torch.eye(a.size(1))
        c = a.unsqueeze(2).expand(*a.size(), a.size(1))
        self.pose_std = torch.clamp(c * b, min=0.0, max=1)

        a = torch.rand(self.state.shape[0], 2)
        b = torch.eye(a.size(1))
        c = a.unsqueeze(2).expand(*a.size(), a.size(1))
        self.goal_std = 3 * torch.clamp(c * b, min=0.0, max=1)

        a = torch.rand(self.state.shape[0], 2)
        b = torch.eye(a.size(1))
        c = a.unsqueeze(2).expand(*a.size(), a.size(1))
        self.velocity_std = torch.clamp(c * b, min=0.0, max=1)

        self.angular_std = torch.clamp(torch.rand(self.state.shape[0], 1), min=0.0, max=1)

    def update_state_std(self, pose_std: torch.Tensor, angular_std: torch.Tensor, vel_std: torch.Tensor):
        """
        :param pose_std: tensor of poses std. expected shape [num_agents, 2, 2]
        :param angular_std: tensor of angular std. expected shape [num_agents, 1, 1]
        :param vel_std: tensor of velocities std. expected shape [num_agents, 2, 2]
        :return:
        """

        self.velocity_std = vel_std
        self.angular_std = angular_std
        self.pose_std = pose_std

    def update_goal(self, goal):
        self.goal = goal

    def update_goal_std(self, std: torch.Tensor):
        self.goal_std = std

    def state_prob(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state: torch tensor with state position [x,y] fot each person
        :return: probability according to mean, and std pose to be at specified pose
        """

        assert state.shape[1] == 2
        assert self.state.shape[0] == self.pose_std.shape[0] == self.goal_std.shape[0] == self.goal.shape[0]
        state_distrib = torch.distributions.multivariate_normal.MultivariateNormal(self.state[:, 0:2],
                                                                                   self.pose_std)
        state_probability = torch.exp(state_distrib.log_prob(state).requires_grad_(True))
        return state_probability

    def angular_prob(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state: torch tensor with heading state [thetta] fot each person
        :return: probability according to mean, and std pose to be at specified heading
        """
        assert state.shape[1] == 1
        assert self.state.shape[0] == self.angular_std.shape[0] == self.goal_std.shape[0] == self.goal.shape[0]
        angular_distrib = torch.distributions.multivariate_normal.MultivariateNormal(self.state[:, 2:3],
                                                                                     self.angular_std.unsqueeze(2))
        angular_probability = torch.exp(angular_distrib.log_prob(state).requires_grad_(True))
        return angular_probability

    def velocity_prob(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param state: torch tensor with velocity state [Vx, Vy] for each person
        :return: probability according to mean, and std pose to be at specified velocity
        """
        assert state.shape[1] == 2
        assert self.state.shape[0] == self.velocity_std.shape[0] == self.goal_std.shape[0] == self.goal.shape[0]
        velocity_distrib = torch.distributions.multivariate_normal.MultivariateNormal(self.state[:, 3:5],
                                                                                      self.velocity_std)
        velocity_probability = torch.exp(velocity_distrib.log_prob(state).requires_grad_(True))
        return velocity_probability

    def goal_prob(self, goal: torch.Tensor) -> torch.Tensor:
        """
        :param goal: torch tensor with velocity state [Vx, Vy] for each person
        :return: probability according to mean, and std pose to be at specified velocity
        """
        assert goal.shape[1] == 2
        assert self.state.shape[0] == self.goal_std.shape[0] == self.goal.shape[0]
        goal_distrib = torch.distributions.multivariate_normal.MultivariateNormal(self.goal[:, 0:2],
                                                                                  self.goal_std)
        goal_probability = torch.exp(goal_distrib.log_prob(goal).requires_grad_(True))
        return goal_probability

    def get_prob(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        :param state: torch tensor with state [x, y, thetta, Vx, Vy] for each person
        :param goal: torch tensor with goal [x,y, thetta] for each person
        :return: probability according to mean, and std state and goal
        """
        goal_prob = self.goal_prob(goal[:, 0:2])
        state_prob = self.state_prob(state[:, 0:2])
        # velocity_prob = self.velocity_prob(state[:, 3:5])
        # angular_prob = self.angular_prob(state[:, 2:3])
        total_prob = goal_prob * state_prob #* velocity_prob * angular_prob
        return total_prob


# measure risks from covariances
# acceleration
# different
# recurent net
