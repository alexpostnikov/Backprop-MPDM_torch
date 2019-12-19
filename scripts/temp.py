import torch
from force_attract_with_dest import force_goal, pose_propagation, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, calc_forces
from Visualizer2 import Visualizer2
from Param import Param
import rospy
import time
from utils import check_poses_not_the_same

lr = 2*10**-5


if __name__ == '__main__':
    rospy.init_node("vis")
    v = Visualizer2()
    param = Param()
    # torch.autograd.set_detect_anomaly(True)
    data = param.input_state.requires_grad_(True)
    goals = param.goal
    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    robot_init_pose = param.robot_init_pose.requires_grad_(True)

    gradient = None
    vv = Visualizer2("v2")
    vvvv = Visualizer2("robot", color=1)
    learning_vis= Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2)
    
    # learning_data.requires_grad_(False)
    for optim_number in range(0,100000):
        
        inner_data = data.clone().detach()
        inner_data.retain_grad()
        vv.publish(goals)
        v.publish(inner_data[1:])
        vvvv.publish(inner_data[0:1])
        
        
        if rospy.is_shutdown():
            break
        
        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
        
        if inner_data.grad is not None:
            inner_data.grad.data.zero_()
        if data.grad is not None:
            data.grad.data.zero_()

        learning_data = data.clone()
        for i in range(0,10):
            
            rf, af = calc_forces(inner_data, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
            F = rf + af
            inner_data = pose_propagation(F, inner_data, param.DT, param.pedestrians_speed)
            temp=calc_cost_function(param.a, param.b, param.e, goals, robot_init_pose, inner_data)
            cost = cost + temp
            learning_data = torch.cat((learning_data,inner_data.clone()))
        learning_vis.publish(learning_data)
        if (optim_number % 1 == 0):
            print ("       ---iter # ",optim_number, "  cost", cost.sum().item())

        cost = cost.sum()
        cost.backward()
        gradient =  inner_data.grad
        gradient[0,:] *= 0
        # print ("gradient:", gradient)

        if gradient is not None:
            with torch.no_grad():
                data[1:,0:2] = data[1:,0:2] + lr * gradient[1:,0:2]
                data[1:,2:4] = data[1:,2:4] + lr   * gradient[1:,2:4]
                data.retain_grad()

        for i in range( data.shape[0]):
            for j in range(i,data.shape[0]):
                # check that they are not in the same place
                if i != j:
                    data[i,0:2], data[j,0:2] = check_poses_not_the_same(data[i,0:2], data[j,0:2], gradient[i,0:2], gradient[j,0:2], lr)


    # ####################
    inner_data = data.clone().detach()
    inner_data.retain_grad()
    # rf, af = calc_forces(inner_data, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
    # F = rf + af
    # inner_data = pose_propagation(F, inner_data, param.DT, param.pedestrians_speed)


    ###### 
    # data = torch.rand((4,4)).requires_grad_(True)
    # data.retain_grad()
    # cost=calc_cost_function(param.a, param.b, param.e, goals, data[0,0:2], data.clone())
    # cost = cost.view(-1,1)
    # cost = cost.sum()
    
    # # cost.backward(data[:,0:1], retain_graph=True)
    # cost.backward()
    # gradient = data.grad
    # print ("gradient:", gradient)
    
    # cost.backward(inner_data[:,0:2])
    # gradient =  inner_data.grad
    # print ("       ---grad:\n", gradient)

    # gradient =  data.grad
    # print ("       ---grad:\n", gradient)
    # inner_data.grad.data.zero_()
    # data.grad.data.zero_()



    # cost.backward(data[:,2:4])
    # print ("grad" , data.grad)
