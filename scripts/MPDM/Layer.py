#! /home/robot/miniconda3/envs/bonsai/bin/python
import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    # input_features, output_features, bias=True):
    def __init__(self, transition_model, covariance_prediction_model=None):
        super(Linear, self).__init__()
        self.tm = transition_model
        self.cpm = covariance_prediction_model

        if self.cpm is None or self.cpm.model is None:
            print("WARN: covariance prediction model not found")

    def forward(self, input):
 
        #  robot_init_pose = inner_data[0, :3]
        # TODO put state(remove) into stacked_state
        # state, cost, stacked_cov, stacked_state, stacked_state_vis, goals, robot_init_pose, policy = input

        state, stacked_state, cost, stacked_cov, stacked_forces, goals, sub_goal, map, resolution, map_origin = input
        robot_init_pose = stacked_state[0][0][:3]
        # state = 1 * input_state
        if map_origin is None:
            map_origin = robot_init_pose[:2].detach().numpy()
        F, fd = self.tm.calc_forces(state, sub_goal, map, resolution, map_origin)
        out = self.tm.pose_propagation(F, state.clone())

        current_cost = self.tm.calc_cost_function(
            goals[0], robot_init_pose, out)
        new_cost = cost + (current_cost.view(-1, 1))

        stacked_state.append(out.clone())
        stacked_forces.append(fd)
        # calculate covariance
        cov = np.zeros((len(state), 2)).tolist() # TODO: fix that
        if self.cpm is not None and self.cpm.model is not None:
            cov = self.cpm.calc_covariance(
                stacked_cov[-1], stacked_state[-2][:, :2].clone().detach().numpy(), stacked_state[-1][:, :2].clone().detach().numpy())
        stacked_cov.append(cov)
        return out, stacked_state, new_cost, stacked_cov, stacked_forces, goals, sub_goal, map, resolution, map_origin

