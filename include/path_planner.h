#ifndef PATH_PLANNER_H
#define PATH_PLANNER_H

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <nav_msgs/Path.h>


#include <cv_bridge/cv_bridge.h>
#include <grid_map_cv/grid_map_cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp> 
#include <limits>
#include <cmath>
#include <string>
#include <utility>
#include <stdexcept>

#include <priority_point.h>
#include <path_simplifier.h>
#include <global_planner.h>
#include <mpdm/Path.h>

// interesting realisation
// https://docs.ros.org/en/diamondback/api/occupancy_grid_utils/html/shortest__path_8cpp_source.html

class PathPlanner
{
public:
  ros::Publisher path_pub;
  PathPlanner()
  {
     ROS_INFO("global planner created");
  }
  bool callbackSrvPath(mpdm::Path::Request &req, mpdm::Path::Response &res)
  {
    // convert to grid map point
    grid_map::Position pStart = {req.start.pose.position.x, req.start.pose.position.y};
    grid_map::Position pGoal = {req.goal.pose.position.x, req.goal.pose.position.y};
    // check cornering cases
    checkOrPutPointOnMap(pStart, grid_map);
    checkOrPutPointOnMap(pGoal, grid_map);
    // getting index into map
    grid_map::Index iStart;
    grid_map::Index iGoal;
    grid_map.getIndex(pStart, iStart);
    grid_map.getIndex(pGoal, iGoal);
    // check goal point on unreachable
    if (grid_map.at(layer, iGoal)){
      // res.path. = -1;
      ROS_WARN_DELAYED_THROTTLE(1,"goal in collision");
      return true;
    } 
    PriorityPoint start = {iStart[0], iStart[1]};
    PriorityPoint goal = {iGoal[0], iGoal[1]};
    ROS_INFO("got new goal");
    std::vector<grid_map::Index> vector_path = makePath_ThetaStar(start, goal, grid_map);
    // std::cout<<vector_path;
    nav_msgs::Path path, simple_path;
    create_ROS_path(path,vector_path, req.goal.pose.orientation, frame_id);
    // FilterRosPath(simple_path, full, 0.1);
    path_pub.publish(path);
    res.path = path;
    return true;
  }
  void callbackMap(const nav_msgs::OccupancyGrid::ConstPtr &msg)
  {
    grid_map::GridMapRosConverter::fromOccupancyGrid(*msg,layer,grid_map);
    // create inflation static layer
    float resolution = (*msg).info.resolution;
    int erosion_size = (int)(map_inflatioon/resolution);
    const char FREE_SPACE = 0;
    const char INFLATION_OBSTACLE = 100;
    
    cv_bridge::CvImage brimage;
    // grid_map::GridMapCvConverter::toImage<unsigned short, 1>(grid_map, layer, CV_16UC1, 0.0, 0.3, cvimage);
    grid_map::GridMapRosConverter::toCvImage(grid_map, layer, sensor_msgs::image_encodings::MONO8, brimage);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                cv::Point(erosion_size, erosion_size));
    cv::Mat cvimage = brimage.image;
    cv::dilate(cvimage, cvimage, element);
    auto imageMsg = cv_bridge::CvImage((*msg).header, sensor_msgs::image_encodings::MONO8, cvimage).toImageMsg();
    grid_map::GridMapRosConverter::addLayerFromImage(*imageMsg, layer, grid_map, FREE_SPACE, INFLATION_OBSTACLE);
    // gridmap["static_inflation_layer"] = gridmap["static_layer"].cwiseMax(gridmap["static_inflation_layer"]);
    ///////////////////////////////////////
    frame_id = (*msg).header.frame_id;
    ROS_INFO("got the map");
  }

private:
  nav_msgs::OccupancyGrid map;
  grid_map::GridMap grid_map;
  std::string layer = "static";
  std::string frame_id;
  float map_inflatioon = 0.25;
  // std::vector<grid_map::Index> path_vector;

  void create_ROS_path(nav_msgs::Path &path_msg, std::vector<grid_map::Index> &path, geometry_msgs::Quaternion &angle, std::string frame) //FIXME: please fix this sheet
  {
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = frame;
    path_msg.poses.clear();
    geometry_msgs::PoseStamped p;
    grid_map::Position pose;
    for (int it = 0; it < path.size(); it++)
    {
      grid_map.getPosition(path[it], pose);
      p.header.stamp = path_msg.header.stamp;
      p.header.frame_id = path_msg.header.frame_id;
      p.pose.position.x = pose.x();
      p.pose.position.y = pose.y();
      p.pose.position.z = 0;
      p.pose.orientation = angle;
      path_msg.poses.push_back(p);
    }
  }

  bool is_points_simular(const grid_map::Position &p1, const grid_map::Position &p2, float min_diff)
  {
    return (fabs(p1[0] - p2[0]) < min_diff) && (fabs(p1[1] - p2[1]) < min_diff);
  }

  bool checkOrPutPointOnMap(grid_map::Position &point, const grid_map::GridMap &map)
  {
    double lengtX = map.getLength().x();
    double lengtY = map.getLength().y();
    double resolution = map.getResolution();
    grid_map::Position temp = point;
    point[0] = point[0] < resolution ? resolution : point[0];
    point[1] = point[1] < resolution ? resolution : point[1];
    point[0] = point[0] > (lengtX - resolution) ? lengtX - resolution : point[0];
    point[1] = point[1] > (lengtY - resolution) ? lengtY - resolution : point[1];
    bool pointOnMap = (point == temp);
    if (!pointOnMap)
    {
      ROS_INFO_THROTTLE(1,"point is out of map");
    }
    return pointOnMap;
  }
};
#endif
