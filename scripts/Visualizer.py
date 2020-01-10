import rospy
import time
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point


class Visualizer:
    def __init__(self, marker_type, topic_name,  marker_size=[1, 1, 1], color=[1, 1, 1, 1], frame_id="/world", in_thread=False, frequency=10):
        self.marker_type = marker_type
        self.topic_name = topic_name
        # TODO: remove govnokod
        if self.marker_type == 'SPHERE' or 'ARROW':
            self.publisher = rospy.Publisher(
                self.topic_name, Marker, queue_size=0)
        elif self.marker_type == 'SPHERES' or 'ARROWS':
            self.publisher = rospy.Publisher(
                self.topic_name, MarkerArray, queue_size=0)

        self.marker_size = marker_size
        self.color = color
        self.frame_id = frame_id
        self.frequency = frequency
        # threading.Thread(target=self.pulishNxgraph_worker, args=(1.0,)).start()
        pass

    # def pulish_worker(self, nxgraph,period):
    #     while not rospy.is_shutdown():
    #         list_points = self.createListNodesPoints(nxgraph)
    #         list_edges = self.createListEdgesPoints(nxgraph)
    #         self.markersPublisher(self.marker_publisher,list_points)
    #         self.ArrowsPublisher(self.arrow_publisher,list_edges)
    #         time.sleep(period)

    def publish(self, data):

        # TODO: remove govnokod

        if self.marker_type == 'SPHERE':
            # got [x,y]
            # or [x,y,z]
            self.markerPublisher(data)

        elif self.marker_type == 'SPHERES':
            # got [[x,y],[x,y],..]
            # or [[x,y,z],[x,y,z],..]
            self.markersPublisher(data)

        elif self.marker_type == 'ARROW':
            # got [x1,y1,x2,y2]
            self.ArrowPublisher(data)

        elif self.marker_type == 'ARROWS':
            # got [[x1,y1,x2,y2],[x1,y1,x2,y2],...]
            self.ArrowsPublisher(data)


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
