
import time
import os
import torch
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection

from statistics import is_inside_sigma, plot_result, calculate_moholonobis

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.stats import multivariate_normal

do_grad = False
do_mc = True

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
            log_folder+=1
        os.mkdir('log/'+str(log_folder))
        w = SummaryWriter('log/'+str(log_folder))
        
        is_inside_mc1 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        is_inside_mc3 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        is_inside_gr3 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        is_inside_gr1 = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        mc1_mah = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        mc3_mah = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        grad1_mah = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        gard3_mah = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[]}
        for batch in range(0, 300):
            self.dataloader.reset_batch_pointer(valid=True)
            x, y, d, numPedsList, PedsList, target_ids = self.dataloader.next_batch()
            self.vp.param.pedestrians_speed = np.linalg.norm(x[0][0][0,1:3] - x[0][5][0,1:3])
            if any(i > 2 for i in numPedsList[0]):
                continue
            if np.linalg.norm(x[0][8][0][1:3] - x[0][0][0][1:3]) < 0.2:
                continue
            print ("new batch")
            grad_cov = [np.array([[1.,0],[0.,1.]]),np.array([[1.,0],[0.,1.]]),np.array([[1.,0],[0.,1.]]),np.array([[1.,0],[0.,1.]])]
            starting_pose = self.dataloader.get_starting_pose(PedsList[0][0:1], x[0][0:1])
            goals_ = self.dataloader.get_ped_goals(PedsList[0], x[0])
            starting_time = self.dataloader.get_starting_time(PedsList[0], x[0])

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
                        self.vp.param.goal = torch.cat((self.vp.param.goal[0:j, :], self.vp.param.goal[1+j:, :]))

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

                    for person in range(0,self.vp.param.input_state.shape[0]):
                        mux = self.vp.param.input_state[person, 0].tolist()
                        muy = self.vp.param.input_state[person, 1].tolist()
                        ellipse = Ellipse((mux, muy),
                            width=np.sqrt(grad_cov[person][0,0])*3,
                            height=np.sqrt(grad_cov[person][1,1])*3, alpha = 0.1
                            # facecolor=colors[ped], ec=colors[ped]
                            # facecolor=facecolor,
                            )
                        ax.add_patch(ellipse)

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
                if do_grad:
                    self.vp.param.input_state.detach()
                    self.vp.param.input_state.requires_grad_(True)
                    rf, af = self.sfm.calc_forces(self.vp.param.input_state, self.vp.param.goal, self.vp.param.pedestrians_speed,
                                        self.vp.param.robot_speed, self.vp.param.k, self.vp.param.alpha, self.vp.param.ped_radius, self.vp.param.ped_mass, self.vp.param.betta)
                    F = rf + af
                    
                    input_state_predicted = self.sfm.pose_propagation(
                        F, self.vp.param.input_state.clone(), self.vp.DT, self.vp.param.pedestrians_speed, self.vp.param.robot_speed)
                    
                    cur_delta_pred = torch.norm(
                        input_state_predicted[:,0:2] - torch.tensor(x[0][i])[:, 1:3], dim=1)
                    mean_cur_delta_pred = torch.mean(cur_delta_pred)
                    mean_cur_delta_pred.backward()

                    grad = self.vp.param.input_state.grad
                    
                    for person_number in range(input_state_predicted.shape[0]):
                        person_grad_cov = np.array( [[grad[person_number,0].detach().numpy(), 0.0],[0.,grad[person_number,1].detach().numpy()]])
                        grad_cov[person_number] = person_grad_cov @ grad_cov[person_number]@ person_grad_cov.T
                        
                        gt_ = torch.tensor(x[0][i])[person_number, 1:3]
                        is_inside_gr1[i-1].append(is_inside_sigma(gt_.tolist(),input_state_predicted[person_number,0:2].tolist(),grad_cov[person_number].tolist(),1))
                        is_inside_gr3[i-1].append(is_inside_sigma(gt_.tolist(),input_state_predicted[person_number,0:2].tolist(),grad_cov[person_number].tolist(),3))
                    
                    self.vp.param.input_state.detach_()
                    self.vp.param.input_state = input_state_predicted.detach()
                    

                
                if do_mc:
                    
                    sampled_poses = []
                    for _ in range(0,20): # for particle
                        
                        input_state_noisy = self.vp.param.input_state.clone()
                        for ped in range(0,input_state_noisy.shape[0]):
                            input_state_noisy[ped,0:2] += multivariate_normal([0,0], grad_cov[ped]).rvs()

                        rf, af = self.sfm.calc_forces(input_state_noisy, self.vp.param.goal, self.vp.param.pedestrians_speed,
                                        self.vp.param.robot_speed, self.vp.param.k, self.vp.param.alpha, self.vp.param.ped_radius, self.vp.param.ped_mass, self.vp.param.betta)
                        F = rf + af
                        
                        input_state_predicted = self.sfm.pose_propagation(
                            F, input_state_noisy.clone(), self.vp.DT, self.vp.param.pedestrians_speed, self.vp.param.robot_speed)
                        
                        cur_delta_pred = torch.norm(
                            input_state_predicted[:,0:2] - torch.tensor(x[0][i])[:, 1:3], dim=1)
                        mean_cur_delta_pred = torch.mean(cur_delta_pred)
                        sampled_poses.append(input_state_predicted)
                    
                    mean_pose = torch.zeros_like(sampled_poses[0])
                    for sampled_pose in sampled_poses:
                        mean_pose += sampled_pose
                    
                    mean_pose = mean_pose/ len(sampled_poses)
                    for ped in range(0,input_state_noisy.shape[0]):
                        grad_cov[ped] = torch.zeros(grad_cov[ped].shape)
                        for sampled_pose in sampled_poses:
                            grad_cov[ped] += (sampled_pose[ped][0:2] - mean_pose[ped][0:2]).view(2,1) @ (sampled_pose[ped][0:2] - mean_pose[ped][0:2]).T.view(1,2)
                        grad_cov[ped] = grad_cov[ped]/ (len(sampled_poses)-1)

                        gt_ = torch.tensor(x[0][i])[ped, 1:3]
                        is_inside_mc1[i-1].append(is_inside_sigma(gt_.tolist(),mean_pose[ped][0:2].tolist(),grad_cov[ped].tolist(),1))
                        is_inside_mc3[i-1].append(is_inside_sigma(gt_.tolist(),mean_pose[ped][0:2].tolist(),grad_cov[ped].tolist(),3))
                        mc1_mah[i-1].append(calculate_moholonobis(gt_.numpy(),mean_pose[ped][0:2].numpy(),grad_cov[ped].numpy()))
                    
                    self.vp.param.input_state = mean_pose

                    # self.vp.param.input_state =  np.mean()

                stacked_trajectories_for_visualizer = torch.cat(
                    (stacked_trajectories_for_visualizer, self.vp.param.input_state.clone()))

                w.add_scalar("cur_averaged_delta", mean_cur_delta_pred, batch*100+i)
                stroka = "\ncur_delta_pred " + str(cur_delta_pred.tolist())
                # print(stroka, end="\r")
                self.save_data.append(stroka)

                
                self.norms.append(mean_cur_delta_pred)
            if self.do_vis:
                plt.close()

        if do_mc:
            correct_pred_mc_3 = np.array([sum(is_inside_mc3[i]) for i in range(12)])
            plot_result(correct_pred_mc_3, np.array([len(is_inside_mc3[0])]*12) , 0.2, title="SFM montecarlo 3 sigma",label="SFM montecarlo 3 sigma")
            np.savez("mc3", gt = correct_pred_mc_3,x = np.array([len(is_inside_mc3[0])]*12), dt=np.array([0.2]))

            correct_pred_mc_1 = np.array([sum(is_inside_mc1[i]) for i in range(12)])
            np.savez("mc1", gt = correct_pred_mc_1,x = np.array([len(is_inside_mc1[0])]*12), dt=np.array([0.2]))
            plot_result(correct_pred_mc_1, np.array([len(is_inside_mc1[0])]*12) , 0.2, title="SFM montecarlo 1 sigma",label="SFM montecarlo 1 sigma")
            np.savez("mc_distance", distance = mc1_mah)

        if do_grad:

            correct_pred_gr_3 = np.array([sum(is_inside_gr3[i]) for i in range(12)])
            plot_result(correct_pred_gr_3, np.array([len(is_inside_gr3[0])]*12) , 0.2, title="SFM forward prop 3 sigma",label="SFM forward prop 3 sigma")
            np.savez("SFM_grad3", gt = correct_pred_gr_3,x = np.array([len(is_inside_gr3[0])]*12), dt=np.array([0.2]))

            correct_pred_gr_1 = np.array([sum(is_inside_gr1[i]) for i in range(12)])
            plot_result(correct_pred_gr_1, np.array([len(is_inside_gr1[0])]*12) , 0.2, title="SFM forward prop 1 sigma",label="SFM forward prop 1 sigma")
            np.savez("SFM_grad1", gt = correct_pred_gr_1,x = np.array([len(is_inside_gr1[0])]*12), dt=np.array([0.2]))

        w.add_scalar("mean_averaged_delta", torch.mean(torch.tensor((self.norms))), 0)
        w.add_scalar("mean_averaged_delta", torch.mean(torch.tensor((self.norms))), 1)

    def print_result(self):
        print(torch.mean(torch.tensor((self.norms))))
        
    def get_result(self):
        return torch.mean(torch.tensor((self.norms)))
        

    def save_result(self, filename = None, data = None):
        from contextlib import redirect_stdout
        if filename is None:
            self.dir = os.path.dirname(os.path.abspath(__file__))
            filename = self.dir + "/result.txt"
        with open(filename, 'w') as file, redirect_stdout(file):
            print(self.save_data)
            print(torch.mean(torch.tensor((self.norms))))
