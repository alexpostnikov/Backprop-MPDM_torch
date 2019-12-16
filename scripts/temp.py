import torch

input_state  = torch.tensor(([2.0,0.5,0.0,-5.0], [1.0,2.5,0.0,0.0]) )
k = 0.5
input_state = input_state.view(-1,4)
goal = torch.tensor(([4.0,1.0], [2.1,2.2]))
goal = goal.view(-1,2)
pedestrians_speed = 1.0

DT = 0.2

def calc_new_pose(input):
    input[:,0:2] = input[:,0:2] + input[:,2:4] * DT
    return input

def calc_new_vel(input_state, forces):
    input_state[:,2:4] = (forces * DT / 60.0 )+ input_state[:,2:4]
    return input_state

t = 0
# plot_data = [[input_state.data[0,0].item()],[input_state.data[0,0].item()],[t]]
plot_data = [[],[],[]]
for i in range(0, 6000):
    

    F_attr = 0.5 * ( pedestrians_speed * ( (goal[:,0:2] - input_state[:,0:2]) / (goal[:,0:2] - input_state[:,0:2]).norm())) - input_state[:,2:4]
    calc_new_vel(input_state, F_attr)
    input_state = calc_new_pose(input_state)
    t +=DT
    
    plot_data[0].append(input_state.data[0,0].item())
    plot_data[1].append(input_state.data[0,1].item())
    plot_data[2].append(t)
# print ((goal[:,0:2] - input_state[:,0:2]).norm())
# print (goal[:,0:2] - input_state[:,0:2])

print (F_attr)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# fig, ax = plt.subplots()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

ax.plot(plot_data[0],  plot_data[1],   plot_data[2],'ro',linewidth=1)
# ax.plot(x2, y2, z,'ro',color='skyblue')
# ax.plot(x3, y3, z,'ro',color='olive')
# ax.plot(x4, y4, z,'ro',color='yellow')
ax.set(zlabel='time (s)', ylabel='y', xlabel = "x",
       title='traj of persons')
ax.grid()

plt.show()

