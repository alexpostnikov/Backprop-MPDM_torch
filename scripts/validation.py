
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

rep_f = RepulsiveForces()
sfm = SFM(rep_f)

param = Param()
vp = ValidationParam(param)

cur_dir = os.path.dirname(os.path.abspath(__file__))
dl = DataLoader(cur_dir+"/Tests", 1, 20, 0)

validator = Validator(validation_param = vp,sfm = sfm, dataloader = dl, do_vis = False)

validator.validate()
validator.save_result(cur_dir+"/validation_result.txt")