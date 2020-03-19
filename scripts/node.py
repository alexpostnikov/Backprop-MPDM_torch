#!/usr/bin/python3
import rospy
from Param import ROS_Param
from Utils.RosPubSub import RosPubSub
from Utils.Visualizer2 import Visualizer2
from MPDM.SFM import SFM
from MPDM.RepulsiveForces import RepulsiveForces
from MPDM.MPDM import MPDM
import time
# from MPDM.Policy import Policy

if __name__ == '__main__':
    rospy.init_node("mpdm")
    ps = RosPubSub()
    # MPDM
    param = ROS_Param()
    rep_f = RepulsiveForces(param)
    sfm = SFM(rep_f, param)
    mpdm = MPDM(param, sfm)
    map = ps.map.update_static_map()
    rospy.sleep(1.0)
    while not (rospy.is_shutdown()):
        robot, path = ps.robot.get_robot_state()
        peds, goals = ps.peds.get_peds_state()
        mpdm.update_state(robot, peds, path[-1], goals, map)
        if mpdm.is_init():
            break
        rospy.sleep(1.0)
        rospy.loginfo("no data of robot_state or peds_state")

    rospy.loginfo("mpdm is initialized")

    while not (rospy.is_shutdown()):
        start = time.time()
        robot, path = ps.robot.get_robot_state()
        peds, goals = ps.peds.get_peds_state()

        mpdm.update_state(robot, peds, path[-1], goals, map)
        mpdm.predict(epoch=10)
        path = mpdm.get_robot_path()

        ps.path.publish(path)
        print("average time: ", time.time() - start)
        # rospy.sleep(0.1)
    exit()
