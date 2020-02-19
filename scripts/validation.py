
import torch
import math
from force_attract_with_dest import force_goal, pose_propagation, is_goal_achieved, generate_new_goal

from Param import Param
from dataloader import DataLoader
from force_attract_with_dest import force_goal, pose_propagation, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, calc_forces

import matplotlib.pyplot as plt


def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for row in list1: 
        for x in row:
            # check if exists in unique_list or not 
            if x not in unique_list: 
                unique_list.append(x) 
    # print list 
#     for x in unique_list: 
#         print (x,)
    return unique_list

def get_ped_goals(pedlist,ped_poses):

    final_goals = {}
    for i in range (len(ped_poses)-1,-1,-1):
        for col in ped_poses[i]:
            if int(col[0]) not in  final_goals.keys():
                    final_goals[int(col[0])] = col[1:3]
    return (final_goals)
                    
                
def get_data(dataloader):
    dataloader.reset_batch_pointer(valid=True)
    x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
    return x, PedsList


def get_starting_time(pedlist,ped_poses):
    ped_nums = unique(pedlist)
    starting_episode = {}
    for i in ped_nums:
        starting_episode[i] = None
    for i,index in enumerate(range(0,len(ped_poses)-1)):
        for col in ped_poses[i]:
            if int(col[0]) in starting_episode.keys():
                if starting_episode[int(col[0])] is None :
                    starting_episode[int(col[0])] = i
    return starting_episode

def get_starting_pose(pedlist,ped_poses):
    ped_nums = unique(pedlist)
    starting_pose = {}
    for i in ped_nums:
        starting_pose[i] = None
    for i,index in enumerate(range(0,len(ped_poses))):
        for col in ped_poses[i]:
            if int(col[0]) in starting_pose.keys():
                if starting_pose[int(col[0])] is None :
                    starting_pose[int(col[0])] = col[1:3]
    return starting_pose

def get_starting_pose_by_id(pedlist,ped_poses, id):
    ped_nums = unique(pedlist)
    starting_pose = []
    for i in range(0,len(ped_poses)-1):
        for col in ped_poses[i]:
            if int(col[0]) == int(id):
                    starting_pose = col[1:3]
                    return starting_pose
    else:
        return None


class Validation_Param(Param):
        
    def __init__(self, numPed):
        super(Validation_Param,self).__init__(num_ped=numPed)
        self.DT = 0.4
        self.socForceRobotPerson = self.socForcePersonPerson
        self.generateMatrices()        
        self.init_calcs(self.device)
        self.robot_goal = self.goal[0,2:4]
        self.index_to_id = {}
        # for i,_ in enumerate(self.input_state):
        #     self.index_to_id[i] = i


    def update_scene(self, new_pose_mean, new_goal_mean):
        self.input_state_mean = new_pose_mean
        if self.device is not None:
            new_pose_mean = new_pose_mean.to(self.device)
        self.input_distrib = torch.distributions.normal.Normal(self.input_state_mean, self.input_state_std)
        self.input_state = self.input_state_mean

        self.goal_mean = new_goal_mean
        
        self.goal_distrib = torch.distributions.normal.Normal(self.goal_mean, self.goal_std)
        self.goal = self.goal_mean
        self.goal = self.goal.view(-1, 2)

    def add_person(self, pose,goal):
        self.input_state_mean = torch.cat((self.input_state_mean, pose),dim=0)
        self.goal = torch.cat((self.goal, goal),dim=0)


do_vis = 1



import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
dataloader = DataLoader(cur_dir,1,20,0)
dataloader.reset_batch_pointer(valid=True)
x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
dataloader.reset_batch_pointer(valid=True)
x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
# import random
# for _ in range(0, int(50*random.random())):
dataloader.reset_batch_pointer(valid=True)
x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()




import time
plt.ion()


norms = []
for _ in range (0,100):
    dataloader.reset_batch_pointer(valid=True)
    x, y, d , numPedsList, PedsList ,target_ids= dataloader.next_batch()
    starting_pose = get_starting_pose(PedsList[0][0:1], x[0][0:1])
    goals_ = get_ped_goals(PedsList[0], x[0])
    starting_time = get_starting_time(PedsList[0], x[0])

    v = Validation_Param(len(starting_pose))

    ped_poses = []
    # print (x[0][0])
    for i,key in enumerate(x[0][0]):
        ped_poses.append( [ x[0][0][i][1], x[0][0][i][2], 0 ,0] )
        v.index_to_id[i] = x[0][0][i][0]

    # print (ped_poses)
    # print (v.index_to_id)
    goals = []
    for idx in range(0,len(ped_poses)):
        goals.append(goals_[v.index_to_id[idx]])


    v.input_state = torch.tensor(ped_poses)
    v.goal = torch.tensor(goals)

    stacked_trajectories_for_visualizer = v.input_state.clone()

    if do_vis:
        fig, ax = plt.subplots(1, 1)    

        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        plt.pause(0.01)
        ax.set_xlabel('distance, m')
        ax.set_ylabel('distance, m')
        
        ax.set_title("prediction visualsiation")
    cur_delta_pred = 0
    print("new set!")
    for i in range(1,20):

        ###### ADD A PERSON (stack to bottom)
        if PedsList[0][i] != list(v.index_to_id.values()):
            for ped_id in PedsList[0][i]:
                if ped_id not in list(v.index_to_id.values()):
                        pose = get_starting_pose(PedsList[0][i:i+1], x[0][i:i+1])[ped_id]
                        v.input_state = torch.cat((v.input_state, torch.tensor([[pose[0], pose[1], 0, 0],])))
                        v.num_ped += 1
                        v.generateMatrices()
                        ped_goal = goals_[ped_id]
                        
                            
                        v.goal = torch.cat((v.goal, torch.tensor([[ped_goal[0],ped_goal[1]] ,],dtype=v.goal.dtype)))
                        
                        v.index_to_id[v.goal.shape[0]-1] = ped_id 
        
        
        # REMOVE PERSONS
        rows_to_remove = []
        for key in v.index_to_id.keys():
            if v.index_to_id[key] not in PedsList[0][i]:
                rows_to_remove.append(key)

        rows_to_remove.sort(reverse = True)
        new_index_to_id={}
        del_counter = len(rows_to_remove)

        for j in range(v.input_state.shape[0]-1,-1,-1):
            if j in rows_to_remove:

                v.input_state = torch.cat((v.input_state[0:j,:], v.input_state[1+j:,:]))
                v.goal = torch.cat((v.goal[0:j,:], v.goal[1+j:,:]))

                del_counter-=1
                v.num_ped -= 1
            else:
                new_index_to_id[j-del_counter] = v.index_to_id[j]
            
        v.index_to_id = new_index_to_id.copy()
        v.generateMatrices()
        # REMOVE PERSONS END

        if do_vis:
            ax.plot(v.input_state[:,0:1].tolist(), v.input_state[:,1:2].tolist(), "g*", markersize = 3, label="predicted")
            ax.plot(torch.tensor(x[0][i-1])[:,1:2].tolist(), torch.tensor(x[0][i-1])[:,2:3].tolist(), "r*", markersize = 3,label="GT")
            ax.grid(True)
            if i==1:
                ax.legend(loc='best', frameon=False)
            ax.set_title("prediction visualsiation\n cur_delta_pred" + str( cur_delta_pred))
            plt.draw()
            plt.show()
            plt.pause(0.1)

        rf, af = calc_forces(v.input_state, v.goal, v.pedestrians_speed, v.robot_speed, v.k, v.alpha, v.ped_radius, v.ped_mass, v.betta)
        F = rf + af
        v.input_state = pose_propagation(F, v.input_state.clone(), v.DT, v.pedestrians_speed, v.robot_speed)
        stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer, v.input_state.clone()))
        cur_delta_pred = torch.norm(v.input_state[:,0:2] - torch.tensor(x[0][i])[:,1:3],dim=1)
        if any(cur_delta_pred) > 2:
            print ("here")
        print ("cur_delta_pred", cur_delta_pred.tolist() , end="\r")
        mean_cur_delta_pred = torch.mean(cur_delta_pred)
        norms.append(mean_cur_delta_pred)
    if do_vis:
        plt.close()

print (torch.mean(torch.tensor((norms))))
