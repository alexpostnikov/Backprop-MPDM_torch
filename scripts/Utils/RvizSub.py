import rospy
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose, Vector3, PointStamped, PoseStamped, PoseWithCovarianceStamped


class RvizSub:
    def __init__(self, goal=[None,None], initialpose=[None,None], topic_PoseStamped='/move_base_simple/goal', topic_PointStamped="/clicked_point", topic_PoseWithCovarianceStamped="/initialpose"):
        self.goal = [None, None]
        self.initialpose = [None, None]
        self.sub_PoseStamped = rospy.Subscriber(
            topic_PoseStamped, PoseStamped, self.callback_goal, goal, 1)
        self.sub_PointStamped = rospy.Subscriber(
            topic_PointStamped, PointStamped, self.callback_goal, goal, 1)
        
        self.sub_PoseWithCovarianceStamped = rospy.Subscriber(
            topic_PoseWithCovarianceStamped, PoseWithCovarianceStamped, self.callback_initialpose, initialpose, 1)

    def callback_goal(self, msg, goal):
        if 'geometry_msgs/PoseStamped' in msg._type:
            goal[0] = msg.pose.position.x
            goal[1] = msg.pose.position.y
        if 'geometry_msgs/PointStamped' in msg._type:
            goal[0] = msg.point.x
            goal[1] = msg.point.y
        self.goal = goal
        print("got new goal: ", goal)

    def callback_initialpose(self, msg, initialpose):
        if 'geometry_msgs/PoseWithCovarianceStamped' in msg._type:
            initialpose[0] = msg.pose.pose.position.x
            initialpose[1] = msg.pose.pose.position.y
        self.initialpose = initialpose
