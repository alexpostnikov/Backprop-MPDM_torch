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
# data = torch.rand((na,4)).requires_grad_(True)
# data = data * 10 - 5
goals = 10 * torch.rand((na,2),requires_grad=False)

# data = torch.tensor(([4.,2.1,0.1,0.21], [40.,21.,0.1,0.0])).requires_grad_(True)
data = torch.tensor(([4., 2., 0.0,-0.0], [40.2, 21.2, -0.0, 0.])).requires_grad_(True)
# data = 1* data
# data.retain_grad()
goals = torch.tensor(([4.1,2.1],[40.,21.])).requires_grad_(False)

cost = torch.zeros((na,2)).requires_grad_(True)





print ("       ---data\n ", data)
print ("       ---robot_pose\n ", robot_pose)
print ("       ---goals\n ", goals)
print ("\n")
gradient = None
for optim_number in range(0,400):
    cost = torch.zeros((na,2)).requires_grad_(True)
    
    if gradient is not None:
        with torch.no_grad():
            # print ("gradient ", gradient[:,2:4])
            # print ("orig data:", data)
            data[:,2:4] =  data[:,2:4] - 10**-1 * gradient[:,2:4]
            # print ("changed data:", data)
            data.retain_grad()
            # print("here?")
        # print ("here")
    
    # print ("data:", data)
    inner_data = data.clone()
    inner_data.retain_grad()
    if inner_data.grad is not None:
        inner_data.grad.data.zero_()
    # .grad.data.zero_()
    if data.grad is not None:
        data.grad.data.zero_()
    for i in range(0,10):
        
        forces = calc_forces(inner_data, goals)
        inner_data = pose_propagation(forces, inner_data)
        temp=calc_cost_function(agents_pose=inner_data[:,0:2])
        cost = cost + temp

    if (optim_number % 5 == 0):
        print ("       ---iter # ",optim_number, "  cost", cost.sum().item())

    cost.backward(inner_data[:,0:2], retain_graph=True)
    gradient =  inner_data.grad
    # print ("       ---grad:\n", gradient)
    # print()
    # print()
    # gradient =  data.grad
    # print ("       ---grad:\n", gradient)
    
    # .grad.data.zero_()
# ####################
print ("data!!", data)
# cost.backward(inner_data[:,0:2])
# gradient =  inner_data.grad
# print ("       ---grad:\n", gradient)

# gradient =  data.grad
# print ("       ---grad:\n", gradient)
# inner_data.grad.data.zero_()
# data.grad.data.zero_()



# cost.backward(data[:,2:4])
# print ("grad" , data.grad)
