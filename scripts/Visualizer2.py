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
        self.arrow_scale = Vector3(0.2,0.2,0.2)
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
                arrow.header.frame_id = self.frame_id
                markerArray.markers.append(arrow)
                id+=1
                f_num+=1
                forces = forces[2:]
        self.publisher.publish(markerArray)

    def createMarker(self, pose, ):
        pass

    def markerPublisher(self, point):
        # TODO: shlopnut` s markersPublisher

        marker = Marker()
        marker.id = -1
        marker.header.frame_id = self.frame_id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = self.marker_size[0]
        marker.scale.y = self.marker_size[1]
        marker.scale.z = self.marker_size[2]
        marker.color.r = self.color[0]
        marker.color.g = self.color[1]
        marker.color.b = self.color[2]
        marker.color.a = self.color[3]
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0
        if len(point) > 2:
            marker.pose.position.z = point[2]
        self.publisher.publish(marker)

    def markersPublisher(self, points):

        markerArray = MarkerArray()
        i = 0
        for point in points:

            marker = Marker()
            marker.id = i
            marker.header.frame_id = self.frame_id
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = self.marker_size[0]
            marker.scale.y = self.marker_size[1]
            marker.scale.z = self.marker_size[2]
            marker.color.r = self.color[0]
            marker.color.g = self.color[1]
            marker.color.b = self.color[2]
            marker.color.a = self.color[3]
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0
            if len(point) > 2:
                marker.pose.position.z = point[2]
            markerArray.markers.append(marker)
            i += 1
        self.publisher.publish(markerArray)

    def ArrowsPublisher(self, points_ab):
        markerArray = MarkerArray()
        i = 0
        for ab in points_ab:
            marker = Marker()
            marker.id = i
            marker.header.frame_id = self.frame_id
            marker.type = marker.ARROW
            marker.action = marker.ADD
            marker.scale.x = self.marker_size[0]
            marker.scale.y = self.marker_size[1]
            marker.scale.z = self.marker_size[2]
            marker.color.r = self.color[0]
            marker.color.g = self.color[1]
            marker.color.b = self.color[2]
            marker.color.a = self.color[3]
            firstPoint = Point()
            firstPoint.x = ab[0]
            firstPoint.y = ab[1]
            firstPoint.z = 0
            marker.points.append(firstPoint)
            secondPoint = Point()
            secondPoint.x = ab[2]
            secondPoint.y = ab[3]
            secondPoint.z = 0
            marker.points.append(secondPoint)
            markerArray.markers.append(marker)
            i += 1

        self.publisher.publish(markerArray)

    def ArrowPublisher(self, point_ab):
        markerArray = MarkerArray()
        i = 0
        marker = Marker()
        marker.id = i
        marker.header.frame_id = self.frame_id
        marker.type = marker.ARROW
        marker.action = marker.ADD
        marker.scale.x = self.marker_size[0]
        marker.scale.y = self.marker_size[1]
        marker.scale.z = self.marker_size[2]
        marker.color.r = self.color[0]
        marker.color.g = self.color[1]
        marker.color.b = self.color[2]
        marker.color.a = self.color[3]
        firstPoint = Point()
        firstPoint.x = point_ab[0]
        firstPoint.y = point_ab[1]
        firstPoint.z = 0
        marker.points.append(firstPoint)
        secondPoint = Point()
        secondPoint.x = point_ab[2]
        secondPoint.y = point_ab[3]
        secondPoint.z = 0
        marker.points.append(secondPoint)
        i += 1

        self.publisher.publish(marker)
