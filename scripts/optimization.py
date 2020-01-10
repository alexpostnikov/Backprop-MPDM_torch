import torch
from force_attract_with_dest import force_goal, pose_propagation, is_goal_achieved, generate_new_goal
from forward import calc_cost_function, calc_forces
from Visualizer2 import Visualizer2
from Param import Param
import rospy
import time
from utils import check_poses_not_the_same

lr = 10**-5
torch.manual_seed(2)


if __name__ == '__main__':
    rospy.init_node("vis")
    pedestrians_visualizer = Visualizer2("peds")
    
    param = Param()
    # torch.autograd.set_detect_anomaly(True)
    
    goals = param.goal
    cost = torch.zeros(param.num_ped, 1).requires_grad_(True)
    observed_state = param.input_state
    robot_init_pose = observed_state[0,0:2]#param.robot_init_pose.requires_grad_(True)

    gradient = None
    ped_goals_visualizer = Visualizer2("goals/ped",size=[0.1, 0.1, 0.5])
    robot_visualizer = Visualizer2("robot", color=1)
    learning_vis= Visualizer2("peds/learning",size=[0.2,0.2,1.0],color=2, with_text = False)

    starting_poses = observed_state.clone()

    for optim_number in range(0,1000):
        start = time.time()
        if rospy.is_shutdown():
            break

        inner_data = starting_poses.clone()
        ped_goals_visualizer.publish(goals)
        pedestrians_visualizer.publish(inner_data[1:])
        robot_visualizer.publish(inner_data[0:1])

        cost = torch.zeros(param.num_ped, 1).requires_grad_(True)

        if inner_data.grad is not None:
            inner_data.grad.data.zero_()

        stacked_trajectories_for_visualizer = starting_poses.clone()
        for i in range(0, int(param.look_ahead_seconds/ param.DT)):

            rf, af = calc_forces(inner_data, goals, param.pedestrians_speed, param.k, param.alpha, param.ped_radius, param.ped_mass, param.betta)
            F = rf + af
            inner_data = pose_propagation(F, inner_data, param.DT, param.pedestrians_speed)
            temp = calc_cost_function(param.a, param.b, param.e, goals, robot_init_pose, inner_data, observed_state)
            cost = cost + temp.view(-1,1)
            stacked_trajectories_for_visualizer = torch.cat((stacked_trajectories_for_visualizer,inner_data.clone()))

        learning_vis.publish(stacked_trajectories_for_visualizer)
        if (optim_number % 1 == 0):

            print ('       ---iter # ',optim_number, "      cost: {:.1f}".format(cost.sum().item()) ,'      iter time: ', "{:.3f}".format(time.time()-start) )

        # print ("cost", cost)
        cost = cost.sum()
        cost.backward()
        gradient =  inner_data.grad
        gradient[0,:] *= 0

        if gradient is not None:
            with torch.no_grad():
                starting_poses[1:,0:2] = starting_poses[1:,0:2] +  torch.clamp(lr *gradient[1:,0:2],max=0.02,min=-0.02)
                starting_poses[1:,2:4] = starting_poses[1:,2:4] +  torch.clamp(1000*lr * gradient[1:,2:4],max=0.02,min=-0.02)
                starting_poses.retain_grad()

        for i in range( starting_poses.shape[0]):
            for j in range(i,starting_poses.shape[0]):
                # check that they are not in the same place
                if i != j:
                    starting_poses[i,0:2], starting_poses[j,0:2] = check_poses_not_the_same(starting_poses[i,0:2], starting_poses[j,0:2], gradient[i,0:2], gradient[j,0:2], lr)

    print ("delta poses:", observed_state[:,0:2] - starting_poses[:,0:2])
