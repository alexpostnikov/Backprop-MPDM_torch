#! /home/robot/miniconda3/envs/bonsai/bin/python
import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    # input_features, output_features, bias=True):
    def __init__(self, hsfm, cpm=None):
        super(Linear, self).__init__()
        self.hsfm = hsfm
        self.cpm = cpm
        if self.cpm.model is None:
            print("WARN: covariance prediction model not found")

    def forward(self, input):
        # TODO put state(remove) into stacked_state
        state, cost, stacked_cov, stacked_state, stacked_state_vis, goals, robot_init_pose, policy = input
        # state = 1 * input_state
        rf, af = self.hsfm.calc_forces(state, goals)
        F = rf + af
        out = self.hsfm.pose_propagation(F, state.clone())
        temp = self.hsfm.calc_cost_function(
            goals[0], robot_init_pose, out, policy)
        new_cost = cost + (temp.view(-1, 1))
        stacked_state_vis = torch.cat(
            (stacked_state_vis, out.clone()))
        stacked_state.append(out.clone())
        # calculate covariance
        cov = np.zeros((len(state), 2)).tolist() # TODO: fix that overengineering
        if self.cpm.model is not None:
            cov = self.cpm.calc_covariance(
                stacked_cov[-1], stacked_state[-2][:, :2].detach().numpy(), stacked_state[-1][:, :2].detach().numpy())
        stacked_cov.append(cov)
        return (out, new_cost, stacked_cov, stacked_state, stacked_state_vis, goals, robot_init_pose, policy)
