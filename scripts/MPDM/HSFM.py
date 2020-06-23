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
        self.Vor = torch.tensor(5.0) # robot_goal_speed

        self.kfi = 4*0.8  # angular control
        self.kfig = 0.05  # angular speed control
        self.alpha = 3.  # coef needed to calculate kfi and kfig
        self.kj = 0.02  # coef needed to calculate kfi and kfig

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
        Ub = torch.zeros_like(forces[:,:2])
        for i in range(len(forces)):
            Ub[i] = self.Kb@(Rots[i].T@forces[i, :2]) #- self.kd * self.Vo

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
        
        Ufi = -self.kfi*diff_angle - self.kfig * ped_angular_speeds  # angular control input to HLM
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
            PG = (robot_pose - robot_init_pose).dot((-robot_init_pose +
                                                     robot_goal) / torch.norm(-robot_init_pose + robot_goal))

        # B = torch.zeros(len(agents_state), 1, requires_grad=False)

        agents_pose = agents_state[:, :2]
        delta = agents_pose - robot_pose[:2] + 1e-6
        norm = -torch.norm(delta, dim=1) / b

        B = torch.exp(norm)  # +0.5
        B = (-a * PG + 1 * B)
        B = B / len(agents_state)
        B = torch.clamp(B, min=0.0002)
        return B

    def calc_forces(self, state, goals):
        rep_force = self.rep_f.calc_rep_forces(state[:, 0:2], state[:, 3:5], param_lambda=1)*3
        # rep_force = torch.zeros_like(rep_force) # TODO: debug string
        attr_force = self.force_goal(state, goals)
        F = rep_force + attr_force
        #  calc force direction
        F[:,2] = self.calc_phi(F[:,:2].clone())
        return F

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
        v_desired_x_y_yaw[:, 0:2] = desired_direction[:,0:2] * self.Vo / norm_direction_lin[:, 0:2]
        
        # print (pedestrians_speed)
        F_attr = k * (v_desired_x_y_yaw - input_state[:, 3:]*self.DT)
        return F_attr
