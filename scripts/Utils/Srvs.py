import rospy

from mpdm.srv import Path,PathRequest,PathResponse
from geometry_msgs.msg import PoseStamped
import numpy as np
class PathSrvSub:
    def __init__(self, service_name="global_planner/Path"):
        self.srv_path = None
        try:
            rospy.wait_for_service(service_name,timeout=3)
            self.srv_path = rospy.ServiceProxy(service_name, Path)
        except:
            rospy.logerr("service "+service_name+" is unavailible")

    def getPath(self,start,goal):
        if self.srv_path == None:
            return None
        req = PathRequest()
        req.start.pose.position.x = goal[0]
        req.start.pose.position.y = goal[1]
        req.goal.pose.position.x = start[0]
        req.goal.pose.position.y = start[1]
        res = self.srv_path(req)
        path = []
        for pose in res.path.poses:
            path.append([pose.pose.position.x,pose.pose.position.y,goal[2]])
        return np.array(path)

# test
if __name__ == '__main__':
    rospy.init_node("mpdm")
    pathSrv = PathSrvSub()
    start = np.array([2,2,0])
    goal = np.array([5,5,1])
    while not (rospy.is_shutdown()):
        path = pathSrv.getPath(start,goal)
        print(path)
        rospy.sleep(1)
    exit()