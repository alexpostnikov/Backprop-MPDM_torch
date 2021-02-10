#!/usr/bin/python3
import rospy
import numpy as np
from Param import ROS_Param
from Utils.RosPubSub import RosPubSub
from Utils.Utils import array_to_ros_path
# from Utils.RosMapTools import map_to_bool_grid_map, get_coordinate_on_map
from MPDM.HSFM import HSFM
from MPDM.MPDM import MPDM
from cov_prediction.SigmaNN import SigmaNN
from MPDM.Policies import SoloPolicy, LeftPolicy, RightPolicy, StopPolicy
from Utils.GlobalPlanner import select_next_goal
import time

if __name__ == '__main__':
    rospy.init_node("mpdm")
    ps = RosPubSub()
    param = ROS_Param()
    trans_model = HSFM(param)  # SFM(param)
    # cov_model = SigmaNN()
    cov_model = None
    policies = []
    policies.append(SoloPolicy())
    # policies.append(LeftPolicy())
    # policies.append(RightPolicy())
    # policies.append(StopPolicy())
    mpdm = MPDM(param, trans_model, cov_model, policies=policies)
    map = ps.map.update_static_map()
    rospy.sleep(1.0)
    while not (rospy.is_shutdown()):
        robot, goal = ps.robot.get_robot_state()
        peds, goals = ps.peds.get_peds_state()
        if robot is not None or goal:
            mpdm.update_state(robot, peds, goal, goals, map)
        if mpdm.is_init():
            break
        rospy.sleep(1.0)
        rospy.loginfo("no data of robot_state or peds_state")
    robot_global_path = [goal]
    rospy.loginfo("mpdm is initialized")
    while not rospy.is_shutdown():

        start = time.time()
        # update state
        try:
            old_goal = goal.copy()
            old_goals = goals.copy()
        except:
            print("goals is None")
        
        robot, goal = ps.robot.get_robot_state()
        peds, goals = ps.peds.get_peds_state()
        
        if robot is None or goal is None or peds is None or goals is None:
            rospy.sleep(0.1)  # for debug
            continue
        
        if (old_goal != goal).any():
            robot_global_path = ps.pathSrv.getPath(robot, goal)
            local_goal= robot_global_path[0]
        try:
            local_goal = select_next_goal(robot, local_goal, robot_global_path)
        except:
            local_goal= goal
        mpdm.update_state(robot, peds, local_goal, goals, map)
        
        # compute
        path_tensor = mpdm.predict(epoch=1)
        # convert to ROS msgs and send out
        ps.path.publish_from_tensor(path_tensor)
        s, g, ct, co, p, pt, f = mpdm.get_learning_data()
        lt = time.time() - start
        # hmm
        g[0] = goal
        # hmm
        ps.learning.publish(s, g, ct, co, p, pt, lt, f)
        # NOTE: return for performance debug
        # print("average time: ", lt)
        rospy.sleep(0.1)  # for debug
    exit()
