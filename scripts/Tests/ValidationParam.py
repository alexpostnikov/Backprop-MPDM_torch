
import torch

class ValidationParam():

    def __init__(self, param):
        self.param = param
        self.DT = 0.4
        self.index_to_id = {}
        # for i,_ in enumerate(self.input_state):
        #     self.index_to_id[i] = i

    def update_scene(self, new_pose_mean, new_goal_mean):
        self.input_state_mean = new_pose_mean
        if self.device is not None:
            new_pose_mean = new_pose_mean.to(self.device)
        self.input_distrib = torch.distributions.normal.Normal(
            self.input_state_mean, self.input_state_std)
        self.input_state = self.input_state_mean

        self.goal_mean = new_goal_mean

        self.goal_distrib = torch.distributions.normal.Normal(
            self.goal_mean, self.goal_std)
        self.goal = self.goal_mean
        self.goal = self.goal.view(-1, 2)

    def update_num_ped(self,num_ped):
        self.param.update_num_ped(num_ped)
        self.index_to_id = {}

    def add_person(self, pose, goal):
        pass
