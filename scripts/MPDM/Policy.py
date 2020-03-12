from optimization import optimize
import math
import torch

class Policy():
    def __init__(self):
        pass
    
    def get_policy_cost(self,sequential, observed_state, goals, param, lr, ped_goals_visualizer,
                  initial_pedestrians_visualizer,
                  pedestrians_visualizer, robot_visualizer,
                  learning_vis, initial_ped_goals_visualizer, policy=None):

    starting_poses = observed_state.clone()
    starting_poses, goals_in_policy, param.robot_speed = apply_policy(
        policy, starting_poses, goals.clone(), param.robot_speed)
    
    goals_in_policy.requires_grad_(True)

    cost = optimize(param.optim_epochs, sequential, starting_poses,
                    param.robot_init_pose, param,
                    goals_in_policy, lr, ped_goals_visualizer,
                    initial_pedestrians_visualizer,
                    pedestrians_visualizer, robot_visualizer,
                    learning_vis, initial_ped_goals_visualizer, policy)
    return cost

    def apply_policy(self,policy, starting_poses, goals, robot_speed):
        if policy == "stop":
            # starting_poses[0,2:4] = -20 * starting_poses[0,2:4].clone()
            robot_speed = 0.0001
            # return state, goals.clone(), robot_speed
        if policy == "right":
            Gx, Gy = rotate([starting_poses[0, 0].data, starting_poses[0, 1].data], [
                            goals[0, 0].data, goals[0, 1].data], -math.pi/4.)
            Gx, Gy = come_to_me(
                [starting_poses[0, 0].data, starting_poses[0, 1].data], [Gx, Gy], 0.5)
            # with torch.no_grad():
            goals[0, 0:2] = torch.tensor(([Gx, Gy]))
            robot_speed = 1.0

            # goals[0,0:2] = torch.tensor(([Gx, Gy ]))
            # return state, goals, robot_speed
        if policy == "left":
            Gx, Gy = rotate([starting_poses[0, 0].data, starting_poses[0, 1].data], [
                            goals[0, 0].data, goals[0, 1].data], math.pi/4.)
            Gx, Gy = come_to_me(
                [starting_poses[0, 0].data, starting_poses[0, 1].data], [Gx, Gy], 0.5)
            # with torch.no_grad():
            goals[0, 0:2] = torch.tensor(([Gx, Gy]))
            robot_speed = 1.0
            # return state, goals, robot_speed

        if policy == "fast":
            robot_speed = 2.0
            # return state, goals.clone(), robot_speed

        if policy == "slow":
            robot_speed = 0.5
            # return state, goals.clone(), robot_speed

        if policy == "go-solo":
            robot_speed = 1.0
        return starting_poses, goals, robot_speed

    
    def come_to_me(self, origin, point,koef):
        ox, oy = origin
        px, py = point

        qx = px+(ox-px)*koef
        qy = py+(oy-py)*koef
        return qx, qy

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

