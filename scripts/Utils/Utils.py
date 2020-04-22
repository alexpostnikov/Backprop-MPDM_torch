import torch
import logging
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Twist


def ps(x, y, yaw=0, frame="map"):
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.orientation.w = 1
    return ps


def p(x, y, yaw=0, frame="map"):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.orientation.w = 1
    return p


def t(x, y, yaw=0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = yaw
    return t

def array_to_ros_path(array, frame_id="map"):
    path = Path()
    path.header.frame_id = frame_id
    for pose in array:
        path.poses.append(ps(pose[0], pose[1], 0, frame_id))
    return path


class Utils:
    def __init__(self):
        pass

    def check_poses_not_the_same(self, pose1, pose2, grad1, grad2, lr):
        counter = 100
        while torch.norm(pose1 - pose2) < 0.6 and counter:
            pose1 = pose1 - lr/4. * grad1
            pose2 = pose2 - lr/4. * grad2
            counter -= 1
        return pose1, pose2

    def setup_logger(self, logger_name, log_file='logs/log.log'):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('logs/log.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(
            "----------------------starting script------------------------------------")
        return logger
        ####### logging init end ######
