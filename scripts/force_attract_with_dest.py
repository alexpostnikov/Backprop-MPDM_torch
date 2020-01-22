import torch

def force_goal(input_state, goal, pedestrians_speed, robot_speed, k):

	v_desired_x_y =  goal[:,0:2]  - input_state[:,0:2]
	v_desired_ = torch.sqrt(v_desired_x_y.clone()[:,0:1]**2 + v_desired_x_y.clone()[:,1:2]**2)	
	v_desired_x_y[1:-1, :] *= pedestrians_speed / v_desired_[1:-1, :]
	v_desired_x_y[0,:] *= robot_speed / v_desired_[0,:]
	# print (pedestrians_speed)
	F_attr = k * (v_desired_x_y - input_state[:,2:4])
	return F_attr

def pose_propagation(force, state, DT, pedestrians_speed):

	vx_vy_uncl = state[:,2:4] + (force*DT)
	dx_dy = state[:,2:4]*DT + (force*(DT**2))*0.5


	# //apply constrains:
	pose_prop_v_unclamped = vx_vy_uncl.norm(dim=1) #torch.sqrt(vx_vy[:,0:1]**2 + vx_vy[:,1:2]**2)
	pose_prop_v = torch.clamp(pose_prop_v_unclamped, max=pedestrians_speed)
	vx_vy = torch.clamp(vx_vy_uncl, max=pedestrians_speed)
	
	dr = dx_dy.norm(dim=1)#torch.sqrt(dx_dy[:,0:1]**2 + dx_dy[:,1:2]**2)
	mask = (pose_prop_v * DT < dr) #* torch.ones(state.shape[0])

	aa = (1.  - (  pose_prop_v * DT) / dr).view(-1,1)
	bb = (dx_dy.t()*mask).t()
	dx_dy = dx_dy.clone()  - ( bb * aa)
	state[:,0:2] = state[:,0:2].clone() + dx_dy

	state[:,2:4] =  vx_vy
	return state




def is_goal_achieved(state, goals):
    is_achieved = state[:,0:2] - goals
    is_achieved = torch.sqrt(is_achieved[:,0]**2 + is_achieved[:,1]**2)
    return is_achieved<0.3

def generate_new_goal(goals, is_achived, input_state,mul = 20):
    
    for i in range (is_achived.shape[0]):
        if is_achived[i].item() == True:
            goals[i,0] = mul*torch.rand(1)
            goals[i,1] = mul*torch.rand(1)
    return goals




if __name__ == "__main__":

	pass
