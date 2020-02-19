
import torch
class SFM:
    def __init__(self, repulsive_forces):
        self.rep_f = repulsive_forces
        pass

    def pose_propagation(self, force, state, DT, pedestrians_speed, robot_speed):

        vx_vy_uncl = state[:, 2:4] + (force*DT)
        dx_dy = state[:, 2:4]*DT + (force*(DT**2))*0.5

        # //apply constrains:
        # torch.sqrt(vx_vy[:,0:1]**2 + vx_vy[:,1:2]**2)
        pose_prop_v_unclamped = vx_vy_uncl.norm(dim=1)
        pose_prop_v = torch.clamp(
            pose_prop_v_unclamped, min=-pedestrians_speed, max=pedestrians_speed)
        pose_prop_v[0] = torch.clamp(
            pose_prop_v_unclamped[0], min=-robot_speed, max=robot_speed)
        vx_vy = torch.clamp(
            vx_vy_uncl, min=-pedestrians_speed, max=pedestrians_speed)
        vx_vy[0, :] = torch.clamp(
            vx_vy_uncl[0, :], min=-robot_speed, max=robot_speed)

        dr = dx_dy.norm(dim=1)  # torch.sqrt(dx_dy[:,0:1]**2 + dx_dy[:,1:2]**2)
        mask = (pose_prop_v * DT < dr)  # * torch.ones(state.shape[0])

        aa = (1. - (pose_prop_v * DT) / dr).view(-1, 1)
        bb = (dx_dy.t()*mask).t()
        dx_dy = dx_dy.clone() - (bb * aa)
        state[:, 0:2] = state[:, 0:2].clone() + dx_dy

        state[:, 2:4] = vx_vy
        return state
    
    
    def calc_forces(self, state, goals, pedestrians_speed, robot_speed, k, alpha, ped_radius, ped_mass, betta, param_lambda = 1):
        rep_force = self.rep_f.calc_rep_forces(state[:, 0:2], alpha, ped_radius, ped_mass, betta, state[:,2:4], param_lambda)
        # rep_force[0] = 0*rep_force[0]
        attr_force = self.force_goal(state, goals, pedestrians_speed,robot_speed, k)
        return rep_force, attr_force


    def force_goal(self, input_state, goal, pedestrians_speed, robot_speed, k):
        v_desired_x_y =  goal[:,0:2]  - input_state[:,0:2]
        v_desired_ = torch.sqrt(v_desired_x_y.clone()[:,0:1]**2 + v_desired_x_y.clone()[:,1:2]**2)	
        v_desired_x_y[1:-1, :] *= pedestrians_speed / v_desired_[1:-1, :]
        v_desired_x_y[0,:] *= robot_speed / v_desired_[0,:]
        # print (pedestrians_speed)
        F_attr = k * (v_desired_x_y - input_state[:,2:4])
        return F_attr