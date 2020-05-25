import rospy
import time
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Pose, Vector3, PointStamped, PoseStamped, PoseWithCovarianceStamped, Quaternion
import numpy as np


class Visualizer4:
    def __init__(self,  topic_name='/visualizer2', frame_id="/world", color=0, size=[0.6, 0.6, 1.8], with_text=True, starting_id=0, mesh_resource=None, mesh_scale=None):
        self.publisher = rospy.Publisher(topic_name, MarkerArray, queue_size=0)
        self.frame_id = frame_id
        self.with_text = with_text
        if mesh_scale is not None:
            size[0] *= mesh_scale
            size[1] *= mesh_scale
            size[2] *= mesh_scale
        self.point_scale = Vector3(size[0], size[1], size[2])
        self.mesh_resource = mesh_resource
        self.text_scale = Vector3(0, 0, (size[0]+size[1]+size[2])/4)
        self.text_color = ColorRGBA(0, 0, 0, 1)
        self.starting_id = starting_id

        self.point_color = ColorRGBA(1, 1, 1, 1)
        self.arrow_scale = Vector3(0.02, 0.1, 0.1)
        self.first_arrow_scale = Vector3(0.08/10., 0.2/10., 0.2/10.)
        self.point_colors = [
            ColorRGBA(1, 1, 1, 1),    # 0 - white
            ColorRGBA(0, 1, 0, 1),    # 1 - green
            ColorRGBA(0, 0, 1, 1),    # 2 - blue
            ColorRGBA(1, 0, 0, 1),    # 3 - red
            ColorRGBA(0, 0, 0, 1)     # 4 - black
        ]
        self.point_color = self.point_colors[color]
        self.arrow_colors = [
            ColorRGBA(0, 1, 0, 1),    # force 1 - green
            ColorRGBA(0, 0, 1, 1),    # force 2 - blue
            ColorRGBA(1, 0, 0, 1),    # force 3 - red
            ColorRGBA(0, 0, 0, 1)     # force 4 - black
        ]
        pass

    def yaw2q(self, yaw):
        return Quaternion(x=0, y=0, z=np.sin(yaw/2), w=np.cos(yaw/2))

    def publish(self, data, text=None):

        # [ [x,y,yaw,x1,y1,yaw1,x2,y2,yaw2,...],
        #   [x,y,yaw,x1,y1,yaw1,x2,y2,yaw2,...]
        #   ...
        # ]
        # x,y - coord of point
        # xn,yn - n forces
        markerArray = MarkerArray()

        id = 0

        # first_point = True
        for n in range(len(data)):
            agent = data[n]
            pose = Pose()
            pose.position.x = agent[0]
            pose.position.y = agent[1]
            pose.position.z = self.point_scale.z/1.5
            pose.orientation = self.yaw2q(agent[2])

            point_marker = Marker(
                id=id,
                type=Marker.SPHERE,
                action=Marker.ADD,
                scale=self.point_scale,
                color=self.point_color,  # 0 - point color
                pose=pose
            )
            if len(agent) < 4:
                point_marker.type = Marker.CUBE
            if self.mesh_resource is not None:
                point_marker.type = Marker.MESH_RESOURCE
                point_marker.mesh_resource = "package://mpdm/resource/mesh/" + \
                    self.mesh_resource  # "robot2.stl"
            point_marker.header.frame_id = self.frame_id
            id += 1
            markerArray.markers.append(point_marker)

            # add some text
            if self.with_text or text:
                text_pose = Pose()
                text_pose.position.x = agent[0]
                text_pose.position.y = agent[1]
                text_pose.position.z = self.point_scale.z+self.text_scale.z/1.7
                text_marker = Marker(
                    id=id,
                    type=Marker.TEXT_VIEW_FACING,
                    action=Marker.ADD,
                    scale=self.text_scale,
                    color=self.text_color,  # 0 - point color
                    pose=text_pose,
                    text=str(n+self.starting_id)
                )
                text_marker.text = str(n+self.starting_id)
                if text:
                    text_marker.text = text
                text_marker.header.frame_id = self.frame_id
                id += 1
                markerArray.markers.append(text_marker)

            forces = agent[3:]
            f_num = 0
            first_arrow = True
            while len(forces > 0):
                first_point = Point(agent[0], agent[1], 0)  # coords of arrow
                second_point = Point(
                    agent[0]+forces[0], agent[1]+forces[1], 0)  # coords of arrow
                arrow = Marker(
                    id=id,
                    type=Marker.ARROW,
                    action=Marker.ADD,
                    scale=self.arrow_scale,
                    color=self.arrow_colors[f_num],  # color of arrow
                    points=[first_point, second_point],
                    colors=[self.arrow_colors[f_num],
                            self.arrow_colors[f_num]]  # color of arrow
                )
                if first_arrow:
                    first_arrow = False
                    arrow.scale = self.first_arrow_scale
                arrow.header.frame_id = self.frame_id
                markerArray.markers.append(arrow)
                id += 1
                f_num += 1
                forces = forces[3:]
        self.publisher.publish(markerArray)
