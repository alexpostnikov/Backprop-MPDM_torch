#!/usr/bin/python3
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import Path
from Utils.Utils import yaw2q
import random
import math
# from tf.transformations import euler_from_quaternion, quartenion_from_euler


def ps(x, y, yaw=0., frame="map"):
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.orientation = yaw2q(yaw)
    return ps


def p(x, y, yaw=0., frame="map"):
    p = Pose()
    p.position.x = x
    p.position.y = y
    p.orientation = yaw2q(yaw)
    return p


def t(x, y, yaw=0):
    t = Twist()
    t.linear.x = x
    t.linear.y = y
    t.angular.z = yaw
    return t

# def generate_new_goal(self, goals, input_state):
#     is_achived = self.is_goal_achieved(input_state, goals)
#     if any(is_achived) == True:
#         for i in range (is_achived.shape[0]):
#             if is_achived[i].item() == True:
#                 goals[i,0] = self.area_size*torch.rand(1)
#                 goals[i,1] = self.area_size*torch.rand(1)
#             if i == 0:
#                 self.robot_init_pose = input_state[i,0:2].clone()
#     self.goal_mean = goals
#     self.goal = goals
#     return goals

# def is_goal_achieved(state, goals):
#     is_achieved = state[:,0:2] - goals
#     is_achieved = torch.sqrt(is_achieved[:,0]**2 + is_achieved[:,1]**2)
#     return is_achieved<0.3

# def generate_new_goals(peds, area = [10,10]):
#     for i in range(0,len(peds),3):

#         position


def distance(p1, p2):
    return ((p1.position.x - p2.position.x)**2+(p1.position.y - p2.position.y)**2)**0.5


def generate_position(area=[10., 10., 3.]):
    p = Pose()
    p.position.x = area[0]*random.random()
    p.position.y = area[1]*random.random()
    p.orientation = yaw2q(area[2]*(random.random()*2.-1.))
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

# TODO: refactor fucking data containers


def callback_pedestrians(msg, vars):
    peds, robot_pose, path, robot_vel = vars
    len_of_batch = int(1 + len(peds.poses)/3)  # 1 - the robot additional state
    # we need to take the second batch with 1st step of interaton
    ped_state_counter = 0
    second_batch_addition_TODO = len_of_batch*2  # TODO: fix that bug
    for i in range(len_of_batch+second_batch_addition_TODO, 2*len_of_batch+second_batch_addition_TODO):
        # update robot state
        if i is len_of_batch+second_batch_addition_TODO:
            # TODO: found why first learning robot position - [i] in the different direction. May be that correlate with initial orientation, that dont changed over time
            robot_pose.pose = msg.markers[i].pose
            # speed = [-msg.markers[i].pose.position.x,
            #     msg.markers[i+1].pose.position.x-msg.markers[i].pose.position.x,
            #     msg.markers[i+1].pose.position.z-msg.markers[i].pose.position.z]
            # robot_vel.linear.x = msg.markers[i+1].pose.position.x
            # robot_vel.linear.y = msg.markers[i+1].pose.position.y
            # robot_vel.angular.z = msg.markers[i+1].pose.orientation.w
            dist_to_goal = distance(
                robot_pose.pose, path.poses[-1].pose)
            if dist_to_goal < 0.3:
                path.poses[-1].pose = generate_position()

            continue
        # TODO: update peds speed
        p1 = msg.markers[i].pose
        p2 = msg.markers[i+len_of_batch].pose
        vel, orient, vorient = get_vov(p1, p2)
        peds.poses[ped_state_counter].position = p1.position
        peds.poses[ped_state_counter].orientation = orient
        peds.poses[ped_state_counter+1].position = vel
        peds.poses[ped_state_counter+1].orientation = vorient
        # speed = [-msg.markers[i].pose.position.x,
        #         msg.markers[i+1].pose.position.x-msg.markers[i].pose.position.x,
        #         msg.markers[i+1].pose.position.z-msg.markers[i].pose.position.z]

        # peds.poses[ped_state_counter+1].position = msg.markers[i+1].pose.position
        # peds.poses[ped_state_counter+1].orientation = msg.markers[i+1].pose.orientation
        # check distance to goal and generate new one
        dist_to_goal = distance(
            peds.poses[ped_state_counter], peds.poses[ped_state_counter+2])
        if dist_to_goal < 0.3:
            peds.poses[ped_state_counter+2] = generate_position()
        ped_state_counter += 3


if __name__ == '__main__':
    rospy.init_node("fake_publicator")
    frame = "map"
    # position
    robot_pose_pub = rospy.Publisher("/odom", PoseStamped, queue_size=1)
    robot_pose = ps(10, 2)
    # velocity
    robot_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    robot_vel = t(0, 0)
    # path(goal)
    robot_path_pub = rospy.Publisher(
        "/move_base/GlobalPlanner/plan", Path, queue_size=1)
    path = Path()
    path.header.frame_id = frame
    path.poses.append(ps(1, 2))
    path.poses.append(ps(2.5, 2.5))
    path.poses.append(ps(5, 2))
    # pedestrians position_velocity_goal
    ped_pub = rospy.Publisher("/peds/pose_vel_goal", PoseArray, queue_size=1)
    peds = PoseArray()
    peds.header.frame_id = frame
    # 1
    # pose[x=2,y=2,yaw=0]
    # vel[vx=0.5,vy=0,vyaw=0]
    # goal[x=0,y=0,yaw=0]
    num_peds = 10
    for i in range(num_peds):
        peds.poses.append(generate_position())
        peds.poses.append(p(0, 0))
        peds.poses.append(generate_position())

    # some subs
    sub1 = rospy.Subscriber("mpdm/learning_with_covariance", MarkerArray,
                            callback_pedestrians, queue_size=1, callback_args=(peds, robot_pose, path, robot_vel))

    while not (rospy.is_shutdown()):
        robot_pose_pub.publish(robot_pose)
        robot_vel_pub.publish(robot_vel)
        robot_path_pub.publish(path)
        ped_pub.publish(peds)
        rospy.sleep(0.5)

    # MPDM
