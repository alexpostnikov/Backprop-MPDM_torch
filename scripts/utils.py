import torch

def check_poses_not_the_same(pose1, pose2, grad1, grad2, lr):
    counter = 100
    while torch.norm(pose1 - pose2) < 0.6 and counter:
        pose1 = pose1 - lr/4. * grad1
        pose2 = pose2 - lr/4. * grad2
        counter -= 1
    return pose1, pose2
