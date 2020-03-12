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
    mpdm = MPDM(param, rep_f, sfm)
    ps.map.update_static_map()
    map = ps.map.static_map
    while not (rospy.is_shutdown()):
        start = time.time()
        robot = ps.robot.get_robot_state()
        peds = ps.peds.get_peds_state()

        mpdm.update_state(robot, peds)
        mpdm.predict()
        path = mpdm.get_robot_path()

        ps.path.publish(path)
        print("average time: ",time.time() - start)
        # rospy.sleep(0.1)
    exit()