import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

import os
p = os.path.abspath('..')
sys.path.insert(0,p)
p = os.path.abspath('.')+"/scripts"
sys.path.insert(0,p)

from Tests.Validator import Validator
from Tests.ValidationParam import ValidationParam
from Tests.DataLoader import DataLoader

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection

from model import update_plot_with_pred_cov


def gen_fake_data():
    x = [list(i for i in range(20))]
    for timestamp in range(20):
        x[0][timestamp] = np.zeros([1,3])
        for ped_number in range(1):
            x_pose = y_pose = ped_number*2
            if timestamp > 0:
                x_pose = x[0][timestamp-1][ped_number][1] - 0.3 - 0.5 * (np.random.rand() - 0.5)
                y_pose = x[0][timestamp-1][ped_number][2]  - 0.5 * (np.random.rand() - 0.5)
            x[0][timestamp][ped_number] = np.array([ped_number,x_pose,y_pose])
    return x

# dataload = DataLoader(p+"/Tests", 1, 20, 0)
model = torch.load("model.pth")
# model.eval()


for _ in range(10):
			ax = plt.subplot(111)
			# dataload.reset_batch_pointer(valid=True)
			x = gen_fake_data()#dataload.next_batch()
			cov_list_x = []
			cov_list_y = []
			mux = []
			muy = []

			cov_ = np.array([0.,0])
			cov_list_x.append(cov_[0])
			cov_list_y.append(cov_[1])
			
			mux.append(x[0][0][0][1])
			muy.append(x[0][0][0][2])

			for timestemp in range(1,19):
				cov_prev  = cov_

				pose_prev = x[0][timestemp-1][0][1:3]
				pose_cur  = x[0][timestemp][0][1:3]
				
				mux.append(x[0][timestemp][0][1])
				muy.append(x[0][timestemp][0][2])
				
				input_ = torch.from_numpy(np.stack((cov_prev, pose_prev,pose_cur))).reshape(6,1).float() 
				cov_ = model(input_.T).detach().numpy()[0]
				cov_list_x.append(cov_[0])
				cov_list_y.append(cov_[1])
			update_plot_with_pred_cov([cov_list_x],[cov_list_y],[mux],[muy],ax)