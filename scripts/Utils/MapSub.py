import rospy

from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap

class MapSub:
    def __init__(self, topic_name="/map", service_name = "/static_map"):
        self.sub_map = rospy.Subscriber(topic_name, OccupancyGrid, self.callback_map)
        self.static_map = None
        self.srv_map = None
        try:
            rospy.wait_for_service(service_name,timeout=3)
            self.srv_map = add_two_ints = rospy.ServiceProxy(service_name, GetMap)
        except:
            rospy.logerr("service "+service_name+" is unavailible")

    def callback_map(self, msg):
        self.static_map = msg

    def update_static_map(self):
        if self.srv_map == None:
            return None
        self.static_map = self.srv_map()
        return self.static_map

if __name__ == '__main__':
    rospy.init_node("mpdm")
    map = MapSub()
    # print(map.static_map)
    while not (rospy.is_shutdown()):
        rospy.loginfo("update is "+str(map.update_static_map()))
        rospy.sleep(1)
    exit()