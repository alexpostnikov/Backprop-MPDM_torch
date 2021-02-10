import torch
from MPDM.RepulsiveForces import RepulsiveForces
import numpy as np
import math


class HSFM:
    def __init__(self, param, DT=0.4):
        self.param = param
        self.rep_f = RepulsiveForces(self.param)
        self.DT = self.param.DT  # self.DT = DT
        # positive constant parameters
        self.kf = 1.0  # forward
        self.ko = 0.2  # ortogonal
        self.kd = torch.zeros(2)  # velocity
        self.kd[1] = 1.0  # check this koef
        self.Kb = torch.zeros(2, 2)  # Body k matrix
        self.Kb[0, 0] = self.kf
        self.Kb[1, 1] = self.ko
        self.Vo = torch.tensor(5.0)  # pedestrians_goal_speed
        self.Vor = torch.tensor(5.0)  # robot_goal_speed

        self.kfi = 4*0.8  # angular control
        self.kfig = 0.05  # angular speed control
        self.alpha = 3.  # coef needed to calculate kfi and kfig
        self.kj = 0.02  # coef needed to calculate kfi and kfig
        self.kw = 30.1  # coef needed to calculate obstacle repulsion force
        self.kw_dist = 0.25 # wall force dist
        self.ped_mass = 50
        self.ped_radius = 0.35
        # inercial moment of ped (m*r^2/2)
        self.I = torch.tensor(self.ped_mass*self.ped_radius**2.*0.5)

        self.A = torch.zeros(2, 2)  # angular koef, x3
        self.A[0, 1] = 1

        # self.Vo = self.param.pedestrians_speed # ? Vo - maybe "the pedestrian’s desire to move with a given velocity vector"

    def force2U(self, forces, state):
        ped_angles = state[:, 2].clone()
        ped_angular_speeds = state[:, 5]
        ped_speeds = state[:, 1:3]
        force_angles = forces[:, 2]
        Rots = torch.zeros(len(state), 2, 2)  # rotation from global to Body
        cos_array = torch.cos(ped_angles)
        sin_array = torch.sin(ped_angles)
        Rots[:, 0, 0] = cos_array[:]
        Rots[:, 0, 1] = -sin_array[:]
        Rots[:, 1, 0] = sin_array[:]
        Rots[:, 1, 1] = cos_array[:]
        # control input to HLM in Body frame
        # Ub = (self.Kb@(Rots.T.permute(2,0,1)@forces[:, :2].T)[0]).T #- self.kd * self.Vo
        Ub = torch.zeros_like(forces[:, :2])
        for i in range(len(forces)):
            Ub[i] = self.Kb@(Rots[i].T@forces[i, :2])  # - self.kd * self.Vo

        # float ang1 = a2 - a1;
        # float ang2 = ((a2 - a1) + 6.28);
        # float ang3 = ((a2 - a1) - 6.28);
        # float ang = fabs(ang1) < fabs(ang2) ? ang1 : ang2;
        # ang = fabs(ang) < fabs(ang3) ? ang : ang3;

        # TODO find out how to made this easer
        ang1 = ped_angles-force_angles
        ang2 = (ped_angles-force_angles) + math.pi*2
        ang3 = (ped_angles-force_angles) - math.pi*2
        diff_angle = torch.zeros_like(ang1)
        for n in range(len(diff_angle)):
            if math.fabs(ang1[n]) < math.fabs(ang2[n]):
                diff_angle[n] = ang1[n]
            else:
                diff_angle[n] = ang2[n]
            if math.fabs(diff_angle[n]) > math.fabs(ang3[n]):
                diff_angle[n] = ang3[n]
        # TODO find out how to made this easer

        Ufi = -self.kfi*diff_angle - self.kfig * \
            ped_angular_speeds  # angular control input to HLM
        Ub[:, 0] = torch.clamp(Ub[:, 0].clone(), min=0)
        return Ub, Ufi, Rots

    def pose_propagation(self, force, state):
        Ub, Ufi, Rots = self.force2U(force, state)

        Vb = (1/self.ped_mass)*Ub  # calc linear velocities in Body frame
        # rotate that velocities into global frame
        # TODO: product calculation issue
        for i in range(len(state)):
            state[i, 3:5] = Rots[i]@Vb[i]
        # state[:, 3:5] = (Rots@Vb.T)[0].T

        b = torch.zeros(len(state))  # inercial moment
        b[:] = 1/self.I  # inercial moment
        state[:, 5] = b*Ufi  # calc angular speed

        state[:, :3] = state[:, :3] + state[:, 3:] * self.DT  # move
        return state

    def calc_cost_function(self, robot_goal, robot_init_pose, agents_state):
        a = self.param.a
        b = self.param.b
        e = self.param.e
        robot_pose = agents_state[0, :3]
        robot_speed = agents_state[0, 3:]
        if torch.norm(robot_init_pose - robot_goal) < 1e-6:
            # torch.ones(robot_pose.shape).requires_grad_(True)
            PG = torch.tensor([0.01])
        else:
            pass
        PG = (robot_pose[:2] - robot_init_pose[:2]).dot((-robot_init_pose[:2] +
                                                         robot_goal[:2]) / torch.norm(robot_pose[:2] - robot_init_pose[:2]))

        # B = torch.zeros(len(agents_state), 1, requires_grad=False)

        agents_pose = agents_state[:, :2]
        delta = agents_pose - robot_pose[:2] + 1e-6
        norm = -torch.norm(delta, dim=1) / b

        B = torch.exp(norm)  # +0.5
        B = (-a * PG + 1 * B)
        B = B / len(agents_state)
        B = torch.clamp(B, min=0.0002)
        return B

    def calc_forces(self, state, goals, maps=None, resolution=None, maps_origin = None):
        rep_force = self.rep_f.calc_rep_forces(
            state[:, 0:2], state[:, 3:5], param_lambda=1)*3
        # rep_force = torch.zeros_like(rep_force) # TODO: debug string
        attr_force = self.force_goal(state, goals)
        wall_force = rep_force*0
        if maps is not None and resolution is not None:
            #  calc force direction
            # map positioned to first agent
            wall_force = self.calc_obstacle_force2(state, maps, resolution, maps_origin)
            # Fo = self.calc_obstacle_force2(state, map, resolution) # map positioned to first agent
            # F = rep_force + attr_force + Fo
        F = rep_force + attr_force + wall_force
        F[:, 2] = self.calc_phi(F[:, :2].clone())
        debug_data = [rep_force.clone().tolist(), attr_force.clone().tolist(), wall_force.clone().tolist()]
        
        return F, debug_data

    def calc_obstacle_force2(self, state, maps, resolution, maps_origin):
        poses = state[:, 0:2].detach().numpy()
        # map_origin = poses[0]
        Fw = torch.zeros(state[:, 0:3].shape)
        # prepare map
        poses_in_map = np.flip(
            ((poses-maps_origin)/resolution + np.asarray([n.shape for n in maps])/2).astype(int),1)# [x,y]==[yp,xp]
        for n in range(len(poses_in_map)):
            # pose_in_map = np.rint((poses[n]-map_origin)/resolution + np.asarray(map.shape)/2).astype(int) # TODO: check this line
            # ignore positions that out of map
            # TODO: finish there
            if (maps[n]>0).any() and (poses_in_map[n] >= 0).all() and (poses_in_map[n] < maps[n].shape).all():
                # if n>0:
                #     print("here")
                xp, yp = self.nearest_nonzero_idx_v2(maps[n], poses_in_map[n, 0], poses_in_map[n, 1])
                diff_pose = torch.tensor((poses_in_map[n] - [xp, yp])*resolution).flip(0) 
                dif_dist = np.sqrt(diff_pose[0]*diff_pose[0]+diff_pose[1]*diff_pose[1])
                Fabs = np.exp(-dif_dist)*self.kw
                if self.kw_dist > dif_dist:
                    Fw[n, 0:2] = diff_pose*(Fabs/dif_dist)
                    Fw[n, 1] = -Fw[n, 1]# y axe in another direction
        return Fw
 
    def calc_obstacle_force(self, state, map, resolution, map_origin):
        poses = state[:, 0:2].detach().numpy()
        # map_origin = poses[0]
        Fw = torch.zeros(state[:, 0:3].shape)
        # prepare map
        poses_in_map = np.flip(
            ((poses-map_origin)/resolution + np.asarray(map.shape)/2).astype(int),
            1)# [x,y]==[yp,xp]
        area = map.shape[0]
        for n in range(len(poses_in_map)):
            # pose_in_map = np.rint((poses[n]-map_origin)/resolution + np.asarray(map.shape)/2).astype(int) # TODO: check this line
            # ignore positions that out of map
            # TODO: finish there
            if (map>0).any() and (poses_in_map[n] >= 0).all() and (poses_in_map[n] < map.shape).all():
                # if n>0:
                #     print("here")
                xp, yp = self.nearest_nonzero_idx_v2(map, poses_in_map[n, 0], poses_in_map[n, 1])
                diff_pose = torch.tensor((poses_in_map[n] - [xp, yp])*resolution).flip(0) 
                dif_dist = np.sqrt(diff_pose[0]*diff_pose[0]+diff_pose[1]*diff_pose[1])
                # check self.kw_dist
                if self.kw_dist < dif_dist:
                    Fabs = np.exp(-dif_dist*dif_dist)*self.kw
                    Fw[n, 0:2] = diff_pose*(Fabs/dif_dist)
                    Fw[n, 1] = -Fw[n, 1]# y axe in another direction
                # Fabs = np.exp(-dif_dist*dif_dist)*self.kw
                # Fw[n, 0:2] = diff_pose*(Fabs/dif_dist)
                # Fw[n, 1] = -Fw[n, 1]# y axe in another direction
        return Fw

        # taken from https://stackoverflow.com/questions/43306291/find-the-nearest-nonzero-element-and-corresponding-index-in-a-2d-numpy-array

    def nearest_nonzero_idx(self, a, x, y):
        idx = np.argwhere(a)

        # If (x,y) itself is also non-zero, we want to avoid those, so delete that
        # But, if we are sure that (x,y) won't be non-zero, skip the next step
        idx = idx[~(idx == [x, y]).all(1)]

        return idx[((idx - [x, y])**2).sum(1).argmin()]

    def nearest_nonzero_idx_v2(self, a, x, y):
        tmp = a[x, y]
        a[x, y] = 0
        r, c = np.nonzero(a)
        a[x, y] = tmp
        min_idx = ((r - x)**2 + (c - y)**2).argmin()
        return r[min_idx], c[min_idx]

    def calc_phi(self, F):
        out = []
        for v in F:
            if v[0] > 0:
                out.append(torch.atan(v[1] / v[0]))
            else:
                if v[1] > 0:
                    if v[0] < 0:
                        out.append(np.pi + torch.atan(v[1] / v[0]))
                    else:
                        out.append(np.pi)
                elif v[1] < 0:
                    if v[0] < 0:
                        out.append(-np.pi + torch.atan(v[1] / v[0]))
                    else:
                        out.append(-np.pi)
                else:
                    out.append(0.0)
        return torch.stack(out)

    def force_goal(self, input_state, goal):
        k = self.param.socForcePersonPerson["k"] * torch.ones(len(input_state))
        k[0] = self.param.socForceRobotPerson["k"]
        k = k.view(-1, 1)

        desired_direction = goal[:, 0:3] - input_state[:, 0:3] + 1e-6
        v_desired_x_y_yaw = torch.zeros_like(desired_direction)
        norm_direction_lin = torch.sqrt(desired_direction[:, 0:1] ** 2 +
                                        desired_direction[:, 1:2] ** 2)
        v_desired_x_y_yaw[:, 0:2] = desired_direction[:,
                                                      0:2] * self.Vo / norm_direction_lin[:, 0:2]

        # print (pedestrians_speed)
        if input_state.shape[1] > 2:
            F_attr = k * (v_desired_x_y_yaw - input_state[:, 3:]*self.DT)
        else:
            # TODO: check calculation for shape(Nx2)
            F_attr = k * (v_desired_x_y_yaw - input_state[:]*self.DT)
        return F_attr
