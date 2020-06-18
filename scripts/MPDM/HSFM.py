import torch
from MPDM.RepulsiveForces import RepulsiveForces


class HSFM:
    def __init__(self, param, DT=0.4):
        self.param = param
        self.rep_f = RepulsiveForces(self.param)
        self.DT = self.param.DT  # self.DT = DT
        # positive constant parameters
        self.kf = 1.  # forward
        self.ko = 0.3  # ortogonal
        self.kd = torch.zeros(2)  # velocity
        self.kd[1] = 5.0  # check this koef
        self.Kb = torch.zeros(2, 2)  # Body k matrix
        self.Kb[0, 0] = self.kf
        self.Kb[1, 1] = self.ko
        self.Vo = torch.tensor(1.0)  # pedestrians_goal_speed

        self.kfi = 0.3  # angular control
        self.kfig = 0.1  # angular speed control
        self.alpha = 3.  # koef needed to calculate kfi and kfig
        self.kj = 0.02  # koef needed to calculate kfi and kfig

        self.ped_mass = 80
        self.ped_radius = 0.4
        # inercial moment of ped (m*r^2/2)
        self.I = torch.tensor(self.ped_mass*self.ped_radius**2.*0.5)

        self.A = torch.zeros(2, 2)  # angular koef, x3
        self.A[0, 1] = 1

        # self.Vo = self.param.pedestrians_speed # ? Vo - maybe "the pedestrian’s desire to move with a given velocity vector"

    def force2U(self, forces, state):
        ped_angles = state[:, 2].clone()
        ped_angular_speeds = state[:, 5].clone()
        ped_speeds = state[:, 1:3].clone()
        force_angles = forces[:, 2].clone()
        Rots = torch.zeros(len(state), 2, 2)  # rotation from global to Body
        cos_array = torch.cos(ped_angles)
        sin_array = torch.sin(ped_angles)
        Rots[:, 0, 0] = cos_array[:]
        Rots[:, 0, 1] = -sin_array[:]
        Rots[:, 1, 0] = sin_array[:]
        Rots[:, 1, 1] = cos_array[:]
        # control input to HLM in Body frame
        # TODO:
        Ub = (self.Kb@Rots@forces[:, :2].T)[0].T.reshape(3, 2) - self.kd * self.Vo
        Ufi = -self.kfi*(ped_angles-force_angles) - self.kfig * ped_angular_speeds  # angular control input to HLM
        return Ub, Ufi, Rots

    def pose_propagation(self, force, state):
        Ub, Ufi, Rots = self.force2U(force, state)

        Vb = (1/self.ped_mass)*Ub  # calc linear velocities in Body frame
        # rotate that velocities into global frame
        state[:, 3:5] = (Rots.T.permute(2, 0, 1)@Vb.T)[0].T
        state[:, :2] = state[:, :2].clone() + state[:, 3:5].clone() * \
            self.DT  # move

        b = torch.zeros(len(state))  # inercial moment
        b[:] = 1/self.I  # inercial moment
        state[:, 5] = b.matmul(Ufi)  # calc angular speed
        state[:, 2] = state[:, 2].clone() + state[:, 5].clone() * \
            self.DT  # update self angle
        return state

    def calc_cost_function(self, robot_goal, robot_init_pose, agents_state):
        a = self.param.a
        b = self.param.b
        e = self.param.e
        robot_pose = agents_state[0, :3].clone()
        robot_speed = agents_state[0, 3:].clone()
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
        rep_force = self.rep_f.calc_rep_forces(
            state[:, 0:2], state[:, 3:5], param_lambda=1)
        attr_force = self.force_goal(state, goals)
        return rep_force, attr_force

    def force_goal(self, input_state, goal):
        num_ped = len(input_state)
        k = self.param.socForcePersonPerson["k"] * torch.ones(num_ped)
        k[0] = self.param.socForceRobotPerson["k"]
        k = k.view(-1, 1)

        ps = self.param.pedestrians_speed
        rs = self.param.robot_speed
        desired_direction = goal[:, 0:3] - input_state[:, 0:3] + 1e-6
        v_desired_x_y_yaw = torch.zeros_like(desired_direction)
        norm_direction_lin = torch.sqrt(desired_direction.clone()[:, 0:1] ** 2 +
                                        desired_direction.clone()[:, 1:2] ** 2)
        norm_direction_rot = desired_direction.clone()[:, 2]

        # v_desired_ = torch.sqrt(v_desired_x_y_yaw.clone()[:, 0]**2+v_desired_x_y_yaw.clone()[:, 1]**2+v_desired_x_y_yaw.clone()[:, 2]**2)
        # torch.sqrt(
        #     v_desired_x_y_yaw.clone()[:, 0]**2 +
        #     v_desired_x_y_yaw.clone()[:, 1]**2 +
        #     v_desired_x_y_yaw.clone()[:, 2]**2)
        v_desired_x_y_yaw[1:, 0:2] = desired_direction[1:,
                                                       0:2] * ps / norm_direction_lin[1:, 0:2]
        v_desired_x_y_yaw[0, 0:2] = desired_direction[0,
                                                      0:2] * ps / norm_direction_lin[0, 0:2]

        # TODO: create param: desired rot speed
        v_desired_x_y_yaw[1:, 2] *= desired_direction[1:,
                                                      2] / norm_direction_rot[1:]
        v_desired_x_y_yaw[0, 2] *= desired_direction[0, 2] / \
            norm_direction_rot[0]
        # print (pedestrians_speed)
        F_attr = k * (v_desired_x_y_yaw - input_state[:, 3:])
        return F_attr
