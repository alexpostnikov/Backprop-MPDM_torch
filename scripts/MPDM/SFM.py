
import torch


class SFM:
    def __init__(self, repulsive_forces, param):
        self.rep_f = repulsive_forces
        self.param = param

    def pose_propagation(self, force, state):
        DT = self.param.DT
        ps = self.param.pedestrians_speed
        rs = self.param.robot_speed
        vx_vy_vyaw_uncl = state[:, 3:] + (force*DT)
        dx_dy_dyaw = state[:, 3:]*DT + (force*(DT**2))*0.5

        # //apply constrains:
        # torch.sqrt(vx_vy[:,0:1]**2 + vx_vy[:,1:2]**2)

        pose_prop_v_unclamped = vx_vy_uncl.norm(dim=1)
        pose_prop_v = torch.clamp(pose_prop_v_unclamped, min=-ps, max=ps)
        pose_prop_v[0] = torch.clamp(pose_prop_v_unclamped[0], min=-rs, max=rs)
        vx_vy = torch.clamp(vx_vy_uncl, min=-ps, max=ps)
        vx_vy[0, :] = torch.clamp(vx_vy_uncl[0, :], min=-rs, max=rs)

        dr = dx_dy.norm(dim=1)  # torch.sqrt(dx_dy[:,0:1]**2 + dx_dy[:,1:2]**2)
        mask = (pose_prop_v * DT < dr)  # * torch.ones(state.shape[0])

        aa = (1. - (pose_prop_v * DT) / dr).view(-1, 1)
        bb = (dx_dy.t()*mask).t()
        dx_dy = dx_dy.clone() - (bb * aa)
        state[:, 0:2] = state[:, 0:2].clone() + dx_dy

        state[:, 2:4] = vx_vy
        return state

    def calc_cost_function(self, goal, init_pose, agents_pose, policy=None):
        a = self.param.a
        b = self.param.b
        e = self.param.e
        robot_pose = agents_pose[0, 0:2].clone()
        robot_speed = agents_pose[0, 2:4].clone()
        PG = (robot_pose - init_pose).dot((-init_pose +
                                           goal[0])/torch.norm(-init_pose+goal[0]))
        # Blame
        # PG = torch.clamp(PG, max = 0.5)
        B = torch.zeros(len(agents_pose), 1, requires_grad=False)
        # if torch.norm(robot_speed) > e:
        agents_speed = agents_pose[:, 0:2]
        delta = agents_speed - robot_pose
        norm = -torch.norm(delta, dim=1)/b
        B = torch.exp(norm)  # +0.5
        # Overall Cost
        # PG = PG/len(agents_pose)
        B = (-a*PG+1*B)
        B = B/len(agents_pose)
        B = torch.clamp(B, min=0.002)
        # if ppp != policy:
        #     print (goal, policy)
        #     ppp = policy
        # print (PG, policy)
        return B

    def calc_forces(self, state, goals):
        rep_force = self.rep_f.calc_rep_forces(
            state[:, 0:2], state[:, 3:5], param_lambda=1)
        # rep_force[0] = 0*rep_force[0]
        attr_force = self.force_goal(state, goals)
        return rep_force, attr_force

    def force_goal(self, input_state, goal):
        num_ped = len(input_state)
        k = self.param.socForcePersonPerson["k"] * torch.ones(num_ped)
        k[0] = self.param.socForceRobotPerson["k"]
        k  = k.view(-1,1)

        ps = self.param.pedestrians_speed
        rs = self.param.robot_speed
        v_desired_x_y_yaw = goal[:, 0:3] - input_state[:, 0:3]
        v_desired_ = torch.sqrt(v_desired_x_y_yaw.clone(
        )[:, 0:1]**2 + v_desired_x_y_yaw.clone()[:, 1:2]**2 + v_desired_x_y_yaw.clone()[:, 2:3]**2)
        v_desired_x_y_yaw[1:] *= ps / v_desired_[1:]
        v_desired_x_y_yaw[0] *= rs / v_desired_[0]
        # print (pedestrians_speed)
        F_attr = k * (v_desired_x_y_yaw - input_state[:, 3:])
        return F_attr