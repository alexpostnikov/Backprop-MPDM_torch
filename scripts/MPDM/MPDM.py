from optimization import Linear
import torch.nn as nn
import torch

class MPDM:
    def __init__(self, param, repulsive_forces, sfm, policys=None):
        self.param = param
        self.rep_f = repulsive_forces
        self.map = None
        self.robot = None
        self.peds = None
        self.goals = None
        self.policys = policys
        self.sfm = sfm
        self.modules = []
        self.path = None
        ###### MODEL CREATING ######
        for i in range(0, self.param.number_of_layers):
            self.modules.append(Linear(self.sfm, self.param))
        self.sequential = nn.Sequential(*self.modules)

    def update_state(self, robot=None, peds=None, goals=None, map=None):
        self.map = map.clone().detach().requires_grad_(False)
        self.robot = robot.clone().detach().requires_grad_(False)
        self.peds = peds.clone().detach().requires_grad_(False)
        self.goals = goals.clone().detach().requires_grad_(False)
        self.states = 5

    def predict(self, epoch=20):
        # TODO: try to work without 0) policys 1) map 2) goals 3) peds
        cost = optimize(epoch)
        return True

    def get_robot_path(self,):
        return self.path

    def optimize(self, epochs):

        for epoch_numb in range(0, epochs):
            max_cost = -math.inf
            inner_data = self.states.clone().detach()  # TODO: why copy and detach twice ?
            inner_data.requires_grad_(True)
            ### FORWARD PASS ####
            cost = torch.zeros(len(self.states-1), 1).requires_grad_(True)

            probability_matrix = get_poses_probability(
                inner_data, self.param.input_distrib)
            goal_prob = get_poses_probability(self.goals, self.param.goal_distrib)
            vel_prob = get_poses_probability(inner_data, self.param.input_distrib, index_X=2, index_Y=3)
            # goal_prob[0] = 1. # what?
            _, cost, stacked_trajectories_for_visualizer, _, _, _, _ = self.sequential(
                (inner_data, cost, stacked_trajectories_for_visualizer, self.goals, param, robot_init_pose))

            # print (goals)
            #### VISUALIZE ####
            # if param.do_visualization and None not in [ped_goals_visualizer, initial_pedestrians_visualizer, pedestrians_visualizer, robot_visualizer, learning_vis, initial_ped_goals_visualizer]:
            #     ped_goals_visualizer.publish(goals)
            #     # initial_pedestrians_visualizer.publish(observed_state)
            #     pedestrians_visualizer.publish(starting_poses[1:])
            #     robot_visualizer.publish(starting_poses[0:1])
            #     learning_vis.publish(stacked_trajectories_for_visualizer)
            #     initial_ped_goals_visualizer.publish(param.goal)

            #### CALC GRAD ####

            prob_cost = cost * (probability_matrix) * (goal_prob) * vel_prob

            prob_cost.sum().backward()
            total_cost = prob_cost.sum().item()
            if total_cost > max_cost:
                max_cost = total_cost
            gradient = inner_data.grad
            gradient[0, :] *= 0

            if gradient is not None:
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
            if starting_poses.grad is not None:
                starting_poses.grad.data.zero_()
        return max_cost

    # def get_poses_probability(self, agents_pose, agents_pose_distrib, index_X=0, index_Y=1):

    #     probability = torch.exp(agents_pose_distrib.log_prob(
    #         agents_pose)) * torch.sqrt(2 * math.pi * agents_pose_distrib.stddev**2)
    #     probability_ = 0.5*(probability[:, index_X] + probability[:, index_Y])
    #     probability_ = probability_.view(-1, 1).requires_grad_(True)
    #     return probability_
