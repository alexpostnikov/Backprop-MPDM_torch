#!/usr/bin/python3
import rospy
import numpy as np
from Param import ROS_Param
from Utils.RosPubSub import RosPubSub
from Utils.Utils import array_to_ros_path
from MPDM.HSFM import HSFM
from MPDM.MPDM import MPDM
from cov_prediction.SigmaNN import SigmaNN
import time

if __name__ == '__main__':
    rospy.init_node("mpdm")
    ps = RosPubSub()
    param = ROS_Param()
    trans_model = HSFM(param) #SFM(param)
    cov_model = SigmaNN()
    mpdm = MPDM(param, trans_model, cov_model)
    map = ps.map.update_static_map()
    rospy.sleep(1.0)
    while not (rospy.is_shutdown()):
        robot, goal = ps.robot.get_robot_state()
        peds, goals = ps.peds.get_peds_state()
        if robot is not None:
            mpdm.update_state(robot, peds, goal, goals, map)
        if mpdm.is_init():
            break
        rospy.sleep(1.0)
        rospy.loginfo("no data of robot_state or peds_state")
    rospy.loginfo("mpdm is initialized")

    while not rospy.is_shutdown():
        start = time.time()
        # update state
        robot, goal = ps.robot.get_robot_state()
        # robot[2:4] = np.random.rand(2)
        robot[0:2] = np.random.rand(2)

        # peds, goals = ps.peds.get_peds_state()
        map = ps.map.update_static_map()
        mpdm.update_state(robot, peds, goal, goals, map)
        # compute
        array_path = mpdm.predict(epoch=10)
        # learning = mpdm.get_debug_data()
        # convert to ROS msgs and send out
        outpath = array_to_ros_path(array_path)
        ps.path.publish(outpath)
        print("average time: ", time.time() - start)
        # rospy.sleep(0.1) # for debug
    exit()
