import torch
import math

class Policy:
    def __init__(self):
        pass

    def apply(self, state: torch.Tensor, goals: torch.Tensor):
        raise NotImplemented

    @property
    def name(self):
        raise NotImplemented

class SoloPolicy(Policy):
    def __init__(self):
        super(SoloPolicy, self).__init__()

    def apply(self, state: torch.Tensor, goals: torch.Tensor):
        return state, goals

    @property
    def name(self):
        return "solo"

class LeftPolicy(Policy):

    def __init__(self):
        super(LeftPolicy, self).__init__()

    def apply(self, state: torch.Tensor, goals: torch.Tensor):
        goal_x, goal_y = rotate([state[0, 0].data, state[0, 1].data], [
            goals[0, 0].data, goals[0, 1].data], math.pi / 4.)
        goal_x, goal_y = come_to_me(
            [state[0, 0].data, state[0, 1].data], [goal_x, goal_y], 0.5)
        goals[0, 0:2] = torch.tensor(([goal_x, goal_y]))
        return state, goals

    @property
    def name(self):
        return "left"

class RightPolicy(Policy):

    def __init__(self):
        super(RightPolicy, self).__init__()

    def apply(self, state: torch.Tensor, goals: torch.Tensor):
        goal_x, goal_y = rotate([state[0, 0].data, state[0, 1].data], [
            goals[0, 0].data, goals[0, 1].data], -math.pi / 4.)
        goal_x, goal_y = come_to_me(
            [state[0, 0].data, state[0, 1].data], [goal_x, goal_y], 0.5)
        goals[0, 0:2] = torch.tensor(([goal_x, goal_y]))
        return state, goals

    @property
    def name(self):
        return "right"


class StopPolicy(Policy):

    def __init__(self):
        super(StopPolicy, self).__init__()

    def apply(self, state: torch.Tensor, goals: torch.Tensor):
        goal_x, goal_y = state[0, 0:2]
        goals[0, 0:2] = torch.tensor(([goal_x, goal_y]))
        return state, goals

    @property
    def name(self):
        return "stop"

def come_to_me(origin, point, koef):
    ox, oy = origin
    px, py = point

    qx = px + (ox - px) * koef
    qy = py + (oy - py) * koef
    return qx, qy

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy




# def apply_policy(policy, starting_poses, goals, robot_speed):
    # if policy == "stop":
    #     # starting_poses[0,2:4] = -20 * starting_poses[0,2:4].clone()
    #     robot_speed = 0.0001
    #     # return state, goals.clone(), robot_speed
    # if policy == "right":
    #     Gx, Gy = rotate([starting_poses[0, 0].data, starting_poses[0, 1].data], [
    #         goals[0, 0].data, goals[0, 1].data], -math.pi / 4.)
    #     Gx, Gy = come_to_me(
    #         [starting_poses[0, 0].data, starting_poses[0, 1].data], [Gx, Gy], 0.5)
    #     # with torch.no_grad():
    #     goals[0, 0:2] = torch.tensor(([Gx, Gy]))
    #     robot_speed = 1.0
    #
    #     # goals[0,0:2] = torch.tensor(([Gx, Gy ]))
    #     # return state, goals, robot_speed
    # if policy == "left":
    #     Gx, Gy = rotate([starting_poses[0, 0].data, starting_poses[0, 1].data], [
    #         goals[0, 0].data, goals[0, 1].data], math.pi / 4.)
    #     Gx, Gy = come_to_me(
    #         [starting_poses[0, 0].data, starting_poses[0, 1].data], [Gx, Gy], 0.5)
    #     # with torch.no_grad():
    #     goals[0, 0:2] = torch.tensor(([Gx, Gy]))
    #     robot_speed = 1.0
    #     # return state, goals, robot_speed
    #
    # if policy == "fast":
    #     robot_speed = 2.0
    #     # return state, goals.clone(), robot_speed
    #
    # if policy == "slow":
    #     robot_speed = 0.5
    #     # return state, goals.clone(), robot_speed
    #
    # if policy == "go-solo":
    #     robot_speed = 1.0
    # return starting_poses, goals, robot_speed