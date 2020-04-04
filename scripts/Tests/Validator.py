
import time
import os
import torch
import math
import sys
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#
import numpy as np
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection


def calc_linear_covariance(x):
    peds = {}
    # find all peds and their poses
    for step in range(len(x)):
        for ped in x[step]:
            if ped[0] not in peds:
                peds[ped[0]] = {"pose": [], "start_step": step}
            peds[ped[0]]["pose"].append(ped[1:])
            peds[ped[0]].update({"end_step": step})
    # find vel
    for ped in peds:
        peds[ped]["pose"] = np.array(peds[ped]["pose"])
        peds[ped]["vel"] = peds[ped]["pose"][1] - peds[ped]["pose"][0]
    # create linear aproximation
    for ped in peds:
        peds[ped]["linear"] = peds[ped]["pose"].copy()
        print(peds[ped]["vel"])
        if peds[ped]["vel"][0]<-0.1:
            print("breakpoint")
        for step in range(peds[ped]["end_step"]+1-peds[ped]["start_step"]):
            peds[ped]["linear"][step] =peds[ped]["pose"][0]+peds[ped]["vel"]*step
        peds[ped]["cov"] = np.abs(peds[ped]["pose"] - peds[ped]["linear"])
    return peds

def plot_cov(ped_cov, isok=3):
    colors = ["b", "r", "g", "y"]
    i = 0
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('distance, m')
    ax.set_ylabel('distance, m')
    ax.set_title("gt covariance")
    ax.grid(True)
    for ped in ped_cov:
        ax.plot(ped_cov[ped]["pose"][:, 0],
                ped_cov[ped]["pose"][:, 1],
                colors[i]+"o", label="input ped "+str(ped))
        ax.plot(ped_cov[ped]["linear"][:, 0],
                ped_cov[ped]["linear"][:, 1],
                colors[i]+"*", label="linear ped "+str(ped))
        
        for n in range(len(ped_cov[ped]["pose"])):
            ax.add_patch(Ellipse(xy=(ped_cov[ped]["pose"][n][0], ped_cov[ped]["pose"][n][1]),
                          width=ped_cov[ped]["cov"][n][0]*isok,
                          height=ped_cov[ped]["cov"][n][1]*isok,
                          alpha=0.1, edgecolor=colors[i], facecolor=colors[i]))    
        i+=1
    ax.legend(loc='best', frameon=False)
    plt.pause(2)
    plt.close(fig)



class Validator():
    def __init__(self, validation_param, sfm, dataloader, do_vis=False):
        self.dataloader = dataloader
        self.sfm = sfm
        self.vp = validation_param
        self.dataloader.reset_batch_pointer(valid=True)
        plt.ion()
        self.norms = []
        self.do_vis = do_vis
        self.save_data = []

    def validate(self):
        self.save_data = []
        self.dataloader.reset_batch_pointer(valid=True)
        log_folder = 1
        while os.path.isdir('log/'+str(log_folder)):
            log_folder += 1
        os.mkdir('log/'+str(log_folder))
        w = SummaryWriter('log/'+str(log_folder))

        for batch in range(0, 300):
            self.dataloader.reset_batch_pointer(valid=True)
            x, y, d, numPedsList, PedsList, target_ids = self.dataloader.next_batch()
            starting_pose = self.dataloader.get_starting_pose(
                PedsList[0][0:1], x[0][0:1])
            goals_ = self.dataloader.get_ped_goals(PedsList[0], x[0])
            starting_time = self.dataloader.get_starting_time(
                PedsList[0], x[0])
            ped_cov = calc_linear_covariance(x[0])
            plot_cov(ped_cov)

            self.vp.update_num_ped(len(starting_pose))

            ped_poses = []
            # print (x[0][0])
            for i, key in enumerate(x[0][0]):
                ped_poses.append([x[0][0][i][1], x[0][0][i][2], 0, 0])
                self.vp.index_to_id[i] = x[0][0][i][0]

            # print (ped_poses)
            # print (v.index_to_id)
            goals = []
            for idx in range(0, len(ped_poses)):
                goals.append(goals_[self.vp.index_to_id[idx]])

            self.vp.param.input_state = torch.tensor(ped_poses)
            self.vp.param.goal = torch.tensor(goals)

            stacked_trajectories_for_visualizer = self.vp.param.input_state.clone()

            if self.do_vis:
                fig, ax = plt.subplots(1, 1)

                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                plt.pause(0.01)
                ax.set_xlabel('distance, m')
                ax.set_ylabel('distance, m')

                ax.set_title("prediction visualsiation")
            cur_delta_pred = 0
            self.save_data.append("new set!")
            for i in range(1, 13):

                # ADD A PERSON (stack to bottom)
                if PedsList[0][i] != list(self.vp.index_to_id.values()):
                    for ped_id in PedsList[0][i]:
                        if ped_id not in list(self.vp.index_to_id.values()):
                            pose = self.dataloader.get_starting_pose(
                                PedsList[0][i:i+1], x[0][i:i+1])[ped_id]
                            self.vp.param.input_state = torch.cat(
                                (self.vp.param.input_state, torch.tensor([[pose[0], pose[1], 0, 0], ])))
                            self.vp.param.num_ped += 1
                            self.vp.param.generateMatrices()
                            ped_goal = goals_[ped_id]

                            self.vp.param.goal = torch.cat((self.vp.param.goal, torch.tensor(
                                [[ped_goal[0], ped_goal[1]], ], dtype=self.vp.param.goal.dtype)))

                            self.vp.index_to_id[self.vp.param.goal.shape[0]-1] = ped_id

                # REMOVE PERSONS
                rows_to_remove = []
                for key in self.vp.index_to_id.keys():
                    if self.vp.index_to_id[key] not in PedsList[0][i]:
                        rows_to_remove.append(key)

                rows_to_remove.sort(reverse=True)
                new_index_to_id = {}
                del_counter = len(rows_to_remove)

                for j in range(self.vp.param.input_state.shape[0]-1, -1, -1):
                    if j in rows_to_remove:

                        self.vp.param.input_state = torch.cat(
                            (self.vp.param.input_state[0:j, :], self.vp.param.input_state[1+j:, :]))
                        self.vp.param.goal = torch.cat(
                            (self.vp.param.goal[0:j, :], self.vp.param.goal[1+j:, :]))

                        del_counter -= 1
                        self.vp.param.num_ped -= 1
                    else:
                        new_index_to_id[j-del_counter] = self.vp.index_to_id[j]

                self.vp.index_to_id = new_index_to_id.copy()
                self.vp.param.generateMatrices()
                # REMOVE PERSONS END

                if self.do_vis:
                    ax.plot(self.vp.param.input_state[:, 0:1].tolist(), self.vp.param.input_state[:, 1:2].tolist(
                    ), "g*", markersize=3, label="predicted")
                    ax.plot(torch.tensor(x[0][i-1])[:, 1:2].tolist(), torch.tensor(
                        x[0][i-1])[:, 2:3].tolist(), "r*", markersize=3, label="GT")
                    ax.grid(True)
                    if i == 1:
                        ax.legend(loc='best', frameon=False)
                    ax.set_title(
                        "prediction visualsiation\n cur_delta_pred" + str(cur_delta_pred))
                    plt.draw()
                    plt.show()
                    plt.pause(0.1)

                rf, af = self.sfm.calc_forces(self.vp.param.input_state, self.vp.param.goal, self.vp.param.pedestrians_speed,
                                              self.vp.param.robot_speed, self.vp.param.k, self.vp.param.alpha, self.vp.param.ped_radius, self.vp.param.ped_mass, self.vp.param.betta)
                F = rf + af
                self.vp.param.input_state = self.sfm.pose_propagation(
                    F, self.vp.param.input_state.clone(), self.vp.DT, self.vp.param.pedestrians_speed, self.vp.param.robot_speed)
                stacked_trajectories_for_visualizer = torch.cat(
                    (stacked_trajectories_for_visualizer, self.vp.param.input_state.clone()))
                cur_delta_pred = torch.norm(
                    self.vp.param.input_state[:, 0:2] - torch.tensor(x[0][i])[:, 1:3], dim=1)
                mean_cur_delta_pred = torch.mean(cur_delta_pred)

                w.add_scalar("cur_averaged_delta",
                             mean_cur_delta_pred, batch*100+i)
                stroka = "\ncur_delta_pred " + str(cur_delta_pred.tolist())
                # print(stroka, end="\r")
                self.save_data.append(stroka)

                self.norms.append(mean_cur_delta_pred)
            if self.do_vis:
                plt.close()
        w.add_scalar("mean_averaged_delta", torch.mean(
            torch.tensor((self.norms))), 0)
        w.add_scalar("mean_averaged_delta", torch.mean(
            torch.tensor((self.norms))), 1)

    def print_result(self):
        print(torch.mean(torch.tensor((self.norms))))

    def get_result(self):
        return torch.mean(torch.tensor((self.norms)))

    def save_result(self, filename=None, data=None):
        from contextlib import redirect_stdout
        if filename is None:
            self.dir = os.path.dirname(os.path.abspath(__file__))
            filename = self.dir + "/result.txt"
        with open(filename, 'w') as file, redirect_stdout(file):
            print(self.save_data)
            print(torch.mean(torch.tensor((self.norms))))
