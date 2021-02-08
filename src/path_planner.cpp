#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <path_planner.h>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "global_planner");
  ros::NodeHandle node;
  ros::Rate loop_rate(30);
  PathPlanner path_planner = PathPlanner();
  // debug
  path_planner.path_pub = node.advertise<nav_msgs::Path>("planner/path", 2);
  // debug
  ros::Subscriber sub1 = node.subscribe("map", 1, &PathPlanner::callbackMap, &path_planner);
  ros::ServiceServer service1 = node.advertiseService("global_planner/Path", &PathPlanner::callbackSrvPath, &path_planner);
  while (node.ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return EXIT_SUCCESS;
}
