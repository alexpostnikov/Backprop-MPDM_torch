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

def update_plot_with_pred_cov(cov_x, cov_y, mux,muy, ax, do_pause=True, isok=1):
	colors = ["b", "r","g","y"]
	for ped in range(0, len(mux)):
		for timestemp in range(1,len(mux[ped])):
			ax.plot( [mux[ped][timestemp], mux[ped][timestemp-1]], [muy[ped][timestemp], muy[ped][timestemp-1]], colors[ped]+"-", markersize=1,label="input")
			ellipse = Ellipse((mux[ped][timestemp], muy[ped][timestemp]),
					width=np.sqrt(cov_x[ped][timestemp])*isok,
					height=np.sqrt(cov_y[ped][timestemp])*isok, alpha = 0.1,
					facecolor=colors[ped], ec=colors[ped]
					# facecolor=facecolor,
					)
			ax.add_patch(ellipse)
	plt.show()
	# if do_pause:plt.pause(0.05)


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
		# print(peds[ped]["vel"])
		if peds[ped]["vel"][0]<-0.1:
			pass
			# print("breakpoint")
		for step in range(peds[ped]["end_step"]+1-peds[ped]["start_step"]):
			peds[ped]["linear"][step] =peds[ped]["pose"][0]+peds[ped]["vel"]*step
		peds[ped]["cov"] = np.abs(peds[ped]["pose"] - peds[ped]["linear"])
	return peds

if __name__ == "__main__":
	
	N, D_in, H, D_out = 1, 6, 100, 2

	model = torch.nn.Sequential(
			torch.nn.Linear(D_in, H),
			torch.nn.ReLU(),
			torch.nn.Linear(H, int(H/2)),
			torch.nn.ReLU(),
			torch.nn.Linear(int(H/2), int(H/4)),
			torch.nn.ReLU(),
			torch.nn.Linear(int(H/4), D_out),
			)

	dataload = DataLoader(p+"/Tests", N, 20, 0)

	loss_fn = torch.nn.MSELoss(reduction='sum')
	learning_rate = 1e-4
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch in range(5):
		mean_loss = []
		for t in range(500):
			dataload.reset_batch_pointer(valid=True)
			x, y, d, numPedsList, PedsList, target_ids = dataload.next_batch()
			
			try:
				ped_cov = calc_linear_covariance(x[0])
			except IndexError:
				continue
			
			for ped in ped_cov.keys():
				for  timestemp in range(ped_cov[ped]["start_step"]+1, ped_cov[ped]["end_step"]):
					try:
						cov_prev  = ped_cov[ped]["cov"][timestemp-1]
						pose_prev = ped_cov[ped]["pose"][timestemp-1]
					
						pose_cur  = ped_cov[ped]["pose"][timestemp]
					except IndexError:
						continue
					pass
					input_ = torch.from_numpy(np.stack((cov_prev, pose_prev,pose_cur))).reshape(6,1).float() 
					y_pred = model(input_.T)
					loss = loss_fn(y_pred, torch.from_numpy(ped_cov[ped]["cov"][timestemp]).float())
					mean_loss.append(loss.item())
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

		
		print ("epoch #",epoch, "  mean loss: ", sum(mean_loss)/len(mean_loss))

		if 0:
			for _ in range(10):
				ax = plt.subplot(111)
				dataload.reset_batch_pointer(valid=True)
				x, y, d, numPedsList, PedsList, target_ids = dataload.next_batch()
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

	torch.save(model, "model.pth")

		
		# input -> x,y, cov_prev,cov_next,speed_x,_speed_y
		
		
		# y_pred = model(x)
		# loss = loss_fn(y_pred, y)
		# print(t, loss.item())
		# optimizer.zero_grad()
		# loss.backward()
		# optimizer.step()






		# with torch.no_grad():
		#     for param in model.parameters():
		#         param.data -= learning_rate * param.grad

	# class Net(nn.Module):

	#     def __init__(self):
	#         super(Net, self).__init__()
	#         # 1 input image channel, 6 output channels, 3x3 square convolution
	#         # kernel
	#         # self.conv1 = nn.Conv2d(1, 6, 3)
	#         # self.conv2 = nn.Conv2d(6, 16, 3)
	#         # an affine operation: y = Wx + b
	#         self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
	#         self.fc2 = nn.Linear(120, 84)
	#         self.fc3 = nn.Linear(84, 10)

	#     def forward(self, x):
	#         # Max pooling over a (2, 2) window
	#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
	#         # If the size is a square you can only specify a single number
	#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
	#         x = x.view(-1, self.num_flat_features(x))
	#         x = F.relu(self.fc1(x))
	#         x = F.relu(self.fc2(x))
	#         x = self.fc3(x)
	#         return x

	#     def num_flat_features(self, x):
	#         size = x.size()[1:]  # all dimensions except the batch dimension
	#         num_features = 1
	#         for s in size:
	#             num_features *= s
	#         return num_features


	# net = Net()
	# print(net)