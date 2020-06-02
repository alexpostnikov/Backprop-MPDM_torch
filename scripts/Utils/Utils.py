import torch
import logging
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Twist, Quaternion
import numpy as np
import math


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

def euler_to_quaternion(yaw, pitch, roll):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return [qx, qy, qz, qw]

def yaw2q(yaw):
    # conver yaw angle to quaternion msg 
    return Quaternion(x=0, y=0, z=np.sin(yaw/2), w=np.cos(yaw/2))

def q2yaw(q):
    # conver quaternion msg to yaw angle
    t3 = +2.0 * (q.w * q.z + q.x * q.y)
    t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    # return math.degrees(math.atan2(t3, t4))
    return math.atan2(t3, t4)


def quaternion_to_euler(x, y, z, w):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.atan2(t0, t1)
        # X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)
        # Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)
        # Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z

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
