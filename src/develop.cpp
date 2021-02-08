#include <ros/ros.h>
#include <queue>

// #include <grid_map_ros/grid_map_ros.hpp>
#include <priority_point.h>
#include <global_planner.h>
#include <vector>

// "path" - the output data of your trajectory
// "start" - start point
// "goal" - goal point 
// "layer" - layer of cost_map that we use
// "cost_map" - multi-layered grid map
// "grid_cost" - cost of free space on cost_map
// "only_cost" - flag for calculate only cost of path. path dont need to modify.
// 
//return value is the cost of path that you create or -1 if path not create

// cost_map::Index - integer point on 2d map. example - {100,200}
// PriorityPoint - integer point on 2d map with priority field.

    // Example to print debug information
    // int i = 12345;
    // //That is analog printf(" my variable = %i ", i)
    // ROS_INFO(" my variable = %i ", i);


double makePath_develop(std::vector<grid_map::Index> &path, PriorityPoint start, PriorityPoint goal, std::string layer, grid_map::GridMap &cost_map, double grid_cost, bool only_cost)
{

    return -1;
}
