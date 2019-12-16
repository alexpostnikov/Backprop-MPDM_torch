import rospy
import time
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose, Vector3


class Visualizer2:
    
    def __init__(self,  topic_name='/visualizer2', frame_id="/world"):
        self.publisher = rospy.Publisher(topic_name, MarkerArray, queue_size=0)
        self.frame_id = frame_id
        self.point_scale = Vector3(0.2,0.2,0.2)
        self.point_color = ColorRGBA(1,1,1,1)
        self.arrow_scale = Vector3(0.02,0.1,0.1)
        self.first_arrow_scale = Vector3(0.08,0.2,0.2)
        self.arrow_colors = [
        ColorRGBA(0,1,0,1),    # force 1 - green
        ColorRGBA(0,0,1,1),    # force 2 - blue
        ColorRGBA(1,0,0,1),    # force 3 - red
        ColorRGBA(0,0,0,1)     # force 4 - black
        ]
        pass


    def publish(self, data):
        # [ [x,y,x1,y1,x2,y2]
        #   [x,y,x1,y1]
        #   [x,y,x1,y1,x2,y2,x3,y3,...]
        # ]
        # x,y - coord of point
        # xn,yn - n forces
        markerArray = MarkerArray()
        id = 0
        for agent in data:
            pose = Pose()
            pose.position.x = agent[0]
            pose.position.y = agent[1]
            pose.orientation.w = 1
            
            point_marker = Marker( 
            id=id,
            type=Marker.SPHERE,
            action=Marker.ADD,
            scale=self.point_scale,
            color=self.point_color, # 0 - point color
            pose = pose
            )
            point_marker.header.frame_id = self.frame_id
            id+=1
            markerArray.markers.append(point_marker)
            forces = agent[2:]
            f_num = 0
            first_arrow = True
            while len(forces>0):
                first_point = Point(agent[0],agent[1],0) # coords of arrow
                second_point = Point(agent[0]+forces[0],agent[1]+forces[1],0)  # coords of arrow
                arrow = Marker( 
                id=id,
                type=Marker.ARROW,
                action=Marker.ADD,
                scale=self.arrow_scale,
                color=self.arrow_colors[f_num], # color of arrow
                points=[first_point,second_point],
                colors=[self.arrow_colors[f_num],self.arrow_colors[f_num]] # color of arrow
                )
                if first_arrow:
                    first_arrow = False
                    arrow.scale = self.first_arrow_scale
                arrow.header.frame_id = self.frame_id
                markerArray.markers.append(arrow)
                id+=1
                f_num+=1
                forces = forces[2:]
        self.publisher.publish(markerArray)

