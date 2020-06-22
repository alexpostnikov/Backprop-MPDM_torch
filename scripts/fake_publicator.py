#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path
from Utils.Utils import yaw2q, q2yaw
from mpdm.msg import Ped, Peds, Learning
import random
import math
import time

def ps(x, y, yaw=0., frame="map"):
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.orientation = yaw2q(yaw)
    return ps


def distance(p1, p2):
    return ((p1.position.x - p2.position.x)**2+(p1.position.y - p2.position.y)**2)**0.5

def generate_position(area=[5., 5., 3.]):
    p = Pose()
    p.position.x = area[0]*random.random()
    p.position.y = area[1]*random.random()
    p.orientation = yaw2q(area[2]*(random.random()*2.-1.))
    lock_timer = time.time()
    return p


def get_vov(p1, p2):
    # velocity
    vel = p2.position
    vel.x += -p1.position.x
    vel.y += -p1.position.y
    # orientation
    yaw = math.acos(vel.x/(vel.x**2+vel.y**2)**0.5)
    if vel.y < 0:
        yaw = -yaw
    orient = yaw2q(yaw)
    # velocity of orientation
    vorient = yaw2q(0)
    return vel, orient, vorient


def callback_update_state(msg, vars):
    # TODO: add shepotky consistency
    peds, robot_pose, peds_pub, robot_pose_pub, pub_robot_goal = vars
    # founding out the best_epoch in learning
    best_epoch = msg.epochs[0]
    for epoch in msg.epochs:
        if best_epoch.cost.data > epoch.cost.data:
            best_epoch = epoch
    # update current state on one step
    peds.peds = best_epoch.steps[1].peds
    # check achievieng goals and update their in that case
    # TODO: lock timer of 2s
    for agent in peds.peds:
        dist = distance(agent.position, agent.goal)
        if dist < 0.5:
            # lock
            agent.goal = generate_position()
            if agent.id.data is "0" or agent.id.data is "robot":
                robot_goal = PoseStamped()
                robot_goal.header.frame_id = agent.header.frame_id
                robot_goal.pose = agent.goal
                pub_robot_goal.publish(robot_goal)
    # founding out robot state and update
    for agent in peds.peds:
        if agent.id.data is "0" or agent.id.data is "robot":
            robot_pose.pose = agent.position
    robot_pose_pub.publish(robot_pose)



if __name__ == '__main__':
    rospy.init_node("fake_publicator")
    frame = "map"
    num_peds = 3

    # position
    robot_pose_pub = rospy.Publisher("/odom", PoseStamped, queue_size=1)
    robot_pose = ps(10, 1)
    peds_pub = rospy.Publisher("mpdm/peds", Peds, queue_size=1)
    peds = Peds()
    for i in range(num_peds):
        ped = Ped()
        ped.header.frame_id = frame
        ped.id.data = str(i+1)
        ped.position = generate_position()
        ped.goal = generate_position()
        p = Pose()
        p.position.x = random.random()
        p.position.y = random.random()
        ped.velocity = p
        peds.peds.append(ped)

    # some subs
    pub_PoseStamped = rospy.Publisher(
        "/move_base_simple/goal", PoseStamped, queue_size=1)

    sub_debug = rospy.Subscriber("mpdm/debug", Learning, callback_update_state,
                                 queue_size=1, callback_args=(peds, robot_pose, peds_pub, robot_pose_pub, pub_PoseStamped))

    while not (rospy.is_shutdown()):
        rospy.sleep(0.1)
        peds_pub.publish(peds)
        robot_pose_pub.publish(robot_pose)
