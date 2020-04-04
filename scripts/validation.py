
import time
import os
import torch
import math
import sys
from Param import Param
from MPDM.SFM import SFM
from MPDM.RepulsiveForces import RepulsiveForces
from Tests.Validator import Validator
from Tests.ValidationParam import ValidationParam
from Tests.DataLoader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
rep_f = RepulsiveForces()
sfm = SFM(rep_f)


param = Param()
param.pedestrians_speed = 1.5
vp = ValidationParam(param)

steps = 10


HP_ped_speed = [2.5]
HP_socForcePersonPerson_k =       [4.9]
HP_socForcePersonPerson_lyambda = [1.]
HP_socForcePersonPerson_A =       [12.]
HP_socForcePersonPerson_B =       [0.64]
HP_socForcePersonPerson_d =       [0.26]



counter = 0
w = SummaryWriter('hp_tuning_1/')
for ped_speed in HP_ped_speed:
    for s_k in HP_socForcePersonPerson_k:
        for s_l in HP_socForcePersonPerson_lyambda:
            for s_A in HP_socForcePersonPerson_A:
                for s_b in HP_socForcePersonPerson_B:
                    for s_d in HP_socForcePersonPerson_d:
                            print ("--------------------- iter number ---------------------")
                            print  (counter, " out of ", steps*2)
                            print ("--------------------- iter number ---------------------")
                            counter+=1

                            
                            cur_dir = os.path.dirname(os.path.abspath(__file__))
                            dl = DataLoader(cur_dir+"/Tests", 1, 20, 0)
                            vp.param.pedestrians_speed = ped_speed
                            vp.param.socForceRobotPerson["k"] = s_k
                            vp.param.socForceRobotPerson["lambda"] = s_l
                            vp.param.socForceRobotPerson["A"] = s_A
                            vp.param.socForceRobotPerson["B"] = s_b
                            vp.param.socForceRobotPerson["d"] = s_d
                            vp.param.socForcePersonPerson = vp.param.socForceRobotPerson
                            vp.param.robot_speed = ped_speed
                            # vp.param.betta = s_betta
                            vp.param.generateMatrices()

                            validator = Validator(validation_param = vp,sfm = sfm, dataloader = dl, do_vis = False)
                            hparams = {
                                "ped_speed": ped_speed,
                                "s_k": s_k,
                                "s_l": s_l,
                                "s_A": s_A,
                                "s_b": s_b,
                                "s_d": s_d,
                            }
                            
                            validator.validate()
                            validator.print_result()
                            validator.save_result(cur_dir+"/validation_result.txt")
                            w.add_hparams(hparams, {'hparam/aw_dist': validator.get_result()})