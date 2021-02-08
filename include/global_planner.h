#ifndef GLOBAL_PLANNER_H
#define GLOBAL_PLANNER_H

#include <ros/ros.h>
#include <cmath>
#include <string>
#include <priority_point.h>
#include <grid_map_ros/grid_map_ros.hpp>

// double makePath_AStar(std::vector<cost_map::Index> &path,PriorityPoint start,PriorityPoint goal,std::string layer,cost_map::CostMap &cost_map, double grid_cost, bool only_cost = false);
std::vector<grid_map::Index> makePath_ThetaStar(PriorityPoint start,PriorityPoint goal,grid_map::GridMap &cost_map);
// double makePath_develop(std::vector<grid_map::Index> &path,PriorityPoint start,PriorityPoint goal,std::string layer,grid_map::GridMap &cost_map, double grid_cost, bool only_cost = false);
bool lineOfSight(int i1, int j1, int i2, int j2, grid_map::GridMap &cost_map);
float HeuristicEvclid(const PriorityPoint &source, const PriorityPoint &target);
unsigned int HeuristicChebishev(const PriorityPoint &source, const PriorityPoint &target);
unsigned int HeuristicManhetten(const PriorityPoint &source, const PriorityPoint &target);
unsigned int HeuristicManhetten(int* source, const PriorityPoint &target);
// double makePath_BoostThetaStar(std::vector<grid_map::Index> &path, PriorityPoint start, PriorityPoint goal, std::string layer, grid_map::GridMap &cost_map, double grid_cost, bool only_cost= false);

float HeuristicEvclid(int* source, const PriorityPoint &target);
bool isNear(PriorityPoint &p1,PriorityPoint &p2,unsigned int dist);

#endif