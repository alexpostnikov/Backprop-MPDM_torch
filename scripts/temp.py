import torch
from force_attract_with_dest import k, force_goal, pose_propagation, DT, pedestrians_speed, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, robot_pose, calc_forces
from Visualizer2 import Visualizer2
import rospy
import time
rospy.init_node("vis")

a = 5
b = 2
e = 0.001
rep_coef = 1.
robot_speed = 1

torch.autograd.set_detect_anomaly(True)
na = 2
data = torch.rand((na,4)).requires_grad_(True)
data = data * 5 -2
goals = 5 * torch.rand((na,2),requires_grad=False) -2

# data = torch.tensor(([4.,2.1,0.1,0.21], [40.,21.,0.1,0.0])).requires_grad_(True)
# data = torch.tensor(([0.5, 1.8, 0.0,-0.0], [1.2, 2.9, -0.0, 0.])).requires_grad_(True)
# data = 1* data
# data.retain_grad()
# goals = torch.tensor(([-5.1,-5.1],[5.,5.])).requires_grad_(False)

cost = torch.zeros((na,2)).requires_grad_(True)
v = Visualizer2()

print ("       ---data\n ", data)
print ("       ---robot_pose\n ", robot_pose)
print ("       ---goals\n ", goals)
print ("\n")
gradient = None
vv = Visualizer2("v2")
for optim_number in range(0,4000):
    if rospy.is_shutdown():
        break
    cost = torch.zeros((na,2)).requires_grad_(True)
    
    if gradient is not None:
        with torch.no_grad():
            # print ("gradient ", gradient)
            # print ("orig data:", data)
            # data =  data.clone() + 10**0 * gradient
            data[:,0:2] =  data[:,0:2] + 10**-2 * gradient[:,0:2]
            data[:,2:4] =  data[:,2:4] + 10**-1 * gradient[:,2:4]
            # print ("changed data:", data)
            data.retain_grad()
    
    # print ("data:", data)
    inner_data = data.clone()
    inner_data.retain_grad()
    if inner_data.grad is not None:
        inner_data.grad.data.zero_()
    # .grad.data.zero_()
    if data.grad is not None:
        data.grad.data.zero_()
    b = torch.tensor(([1.,2.01,0,0],))
    plot_data = torch.cat( (inner_data,  b),0)
    
    vv.publish(goals)
    
    v.publish(plot_data)
    start = time.time()
    forces_ar = []
    pose_propagation_ar = []
    calc_cost_function_ar = []
    for i in range(0,10):
        
        st  = time.time()
        forces = calc_forces(inner_data, goals)
        forces_ar.append(time.time() - st)
        st  = time.time()
        inner_data = pose_propagation(forces[0] + forces[1], inner_data, DT=DT, pedestrians_speed=pedestrians_speed)
        pose_propagation_ar.append(time.time() - st)
        st  = time.time()
        temp=calc_cost_function(agents_pose=inner_data[:,0:2])
        calc_cost_function_ar.append(time.time() - st)
        cost = cost + temp
    # print ("forward:", time.time() - start)
    if (optim_number % 25 == 0):
        print ("       ---iter # ",optim_number, "  cost", cost.sum().item())

    start = time.time()
    cost.backward(inner_data[:,0:2], retain_graph=True)
    # print ("backward:", time.time() - start)
    gradient =  inner_data.grad
    # print ("       ---grad:\n", gradient)
    # print()
    # print()
    # gradient =  data.grad
    # print ("       ---grad:\n", gradient)
    
    # .grad.data.zero_()
# ####################
print ("data!!", data)
print ("forces: ", sum(forces_ar)/len(forces_ar))
print ("pose: ", sum(pose_propagation_ar)/len(pose_propagation_ar))
print ("calc_cost_function_ar: ", sum(calc_cost_function_ar)/len(calc_cost_function_ar))
# cost.backward(inner_data[:,0:2])
# gradient =  inner_data.grad
# print ("       ---grad:\n", gradient)

# gradient =  data.grad
# print ("       ---grad:\n", gradient)
# inner_data.grad.data.zero_()
# data.grad.data.zero_()



# cost.backward(data[:,2:4])
# print ("grad" , data.grad)
