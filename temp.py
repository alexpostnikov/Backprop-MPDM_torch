import torch
from force_attract_with_dest import k, force_goal, pose_propagation, DT, pedestrians_speed, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, robot_pose, calc_forces

a = 5
b = 2
e = 0.001
rep_coef = 1.
robot_speed = 1

torch.autograd.set_detect_anomaly(True)
na = 2 
data = torch.rand((na,4)).requires_grad_(True)
data = data * 10 - 5
# data.retain_grad()
cost = torch.zeros((na,2)).requires_grad_(True)


goals = 10 * torch.rand((na,2),requires_grad=False)


print ("       ---data\n ", data)
print ("       ---robot_pose\n ", robot_pose)
print ("       ---goals\n ", goals)
print ("\n")
gradient = None
# for optim_number in range(0,10):
#     cost = torch.zeros((na,2)).requires_grad_(True)
#     if gradient is not None:
#         with torch.no_grad():
#             data[2:4] +=  10**-4 * gradient[2:4]
#             data.retain_grad()
#         print ("here")

#     inner_data = data.clone()
#     inner_data.retain_grad()
inner_data = data.clone()
# inner_data.retain_grad()
for i in range(0,10):
    
    forces = calc_forces(inner_data, goals)
    inner_data = pose_propagation(forces, inner_data)
    temp=calc_cost_function(agents_pose=inner_data[:,0:2])
    cost = cost + temp

print ("       ---cost\n", cost)
cost.backward(inner_data[:,0:2])
gradient =  inner_data.grad
print ("       ---grad:\n", gradient)
inner_data.grad.data.zero_()
data.grad.data.zero_()
# cost.backward(data[:,2:4])
# print ("grad" , data.grad)
