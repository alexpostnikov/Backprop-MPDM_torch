#include <ros/ros.h>
#include <queue>

#include <grid_map_ros/grid_map_ros.hpp>
#include <priority_point.h>
#include <global_planner.h>
#include <vector>
#include <math.h>

//////////////////////////////////////////////////////////////////////////////
//////////////////////////// Theta* //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// ????????????
PriorityPoint neighbors[8];
PriorityPoint current;
// double cost_so_far[300][200];
// int come_from[300][200][2];
// ????????????


std::vector<grid_map::Index> makePath_ThetaStar(PriorityPoint start, PriorityPoint goal, grid_map::GridMap &grid_map)
{
  std::string layer = "static";
  std::priority_queue<PriorityPoint, std::vector<PriorityPoint>, myCompare> openSet;
  size_t max_x = grid_map.getSize()[0];
  size_t max_y = grid_map.getSize()[1];
  double *cost_so_far = new double[max_x*max_y];
  int *come_from = new int[max_x*max_y*2];

  std::fill_n(cost_so_far, max_x * max_y, std::numeric_limits<double>::max());
  cost_so_far[start.x*max_y+start.y] = 0;
  come_from[start.x*max_y*2+start.y*2+0] = start.x;
  come_from[start.x*max_y*2+start.y*2+1] = start.y;
  openSet.push(start);
  // grid_cost=5;
  while (!openSet.empty())
  {
    current = openSet.top();
    openSet.pop();
    if (current == goal)
    {
      break;
    }
    current.GetNeighbors(neighbors);
    double current_cost = cost_so_far[current.x*max_y+current.y];
    // TODO: problem is somewhere here 
    int parent_carent[2] ={come_from[current.x*max_y*2+current.y*2+0], come_from[current.x*max_y*2+current.y*2+1]};
    for (int i = 0; i < 8; i++)
    {
      // TODO: put there into GetNeighbors()
      if (!neighbors[i].OnMap(max_x, max_y))
      {
        continue;
      }
      bool onLine = lineOfSight(parent_carent[0], parent_carent[1], neighbors[i].x, neighbors[i].y, grid_map);
      if (onLine)
      {
        double new_cost = cost_so_far[parent_carent[0]*max_y+parent_carent[1]] + HeuristicEvclid(parent_carent, neighbors[i]);
        if (new_cost < cost_so_far[neighbors[i].x*max_y+neighbors[i].y])
        {
          cost_so_far[neighbors[i].x*max_y+neighbors[i].y] = new_cost;
	        neighbors[i].priority = HeuristicEvclid(neighbors[i], goal) + new_cost;
          openSet.push(neighbors[i]);
          come_from[neighbors[i].x*max_y*2+neighbors[i].y*2+0] = parent_carent[0];
          come_from[neighbors[i].x*max_y*2+neighbors[i].y*2+1] = parent_carent[1];
        }
      }
      else
      {
        double neighbor_price = grid_map.at(layer, grid_map::Index({neighbors[i].x, neighbors[i].y})) + neighbors[i].priority;
        double new_cost = current_cost + neighbor_price;
        if (new_cost < cost_so_far[neighbors[i].x*max_y+neighbors[i].y])
        {
          cost_so_far[neighbors[i].x*max_y+neighbors[i].y] = new_cost;
          neighbors[i].priority =HeuristicEvclid(neighbors[i], goal) + new_cost;
          openSet.push(neighbors[i]);
          come_from[neighbors[i].x*max_y*2+neighbors[i].y*2+0] = current.x;
          come_from[neighbors[i].x*max_y*2+neighbors[i].y*2+1] = current.y;
        }
      }
    }
  }

  // path.clear();
  std::vector<grid_map::Index> path;
  PriorityPoint temp_point;
  
  while (current != start)
  {
    path.push_back({current.x, current.y});
    temp_point.x = come_from[current.x*max_y*2+current.y*2+0];
    temp_point.y = come_from[current.x*max_y*2+current.y*2+1];
    printf("%d %d",temp_point.x,temp_point.y);
    // ROS_INFO(std::to_string(temp_point.x));
    // ROS_INFO(std::to_string(temp_point.y));
    current = temp_point;
  }
  path.push_back({current.x, current.y});
  delete[] cost_so_far;
  delete[] come_from;

  return path;
}


bool lineOfSight(int i1, int j1, int i2, int j2, grid_map::GridMap &grid_map)
{
  grid_map::Index p1 = {i1, j1};
  grid_map::Index p2 = {i2, j2};
  std::string layer = "static";
  for (grid_map::LineIterator iterator(grid_map, p1, p2); !iterator.isPastEnd(); ++iterator)
  {
    if (grid_map.at(layer, *iterator))
    {
      return false;
    }
  }
  return true;
}


//////////////////////////////////////////////////////////////////////////////
////////////////////////////// support ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
bool isNear(PriorityPoint &p1,PriorityPoint &p2,unsigned int dist)
{
  unsigned int a = abs(p1.x - p2.x);
  unsigned int b = abs(p1.y - p2.y);
  return (a + b)<dist;

}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////// Heuristics //////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
float HeuristicEvclid(const PriorityPoint &source, const PriorityPoint &target)
{
  return sqrt(pow(source.x - target.x, 2) + pow(source.y - target.y, 2));
}

float HeuristicEvclid(int* source, const PriorityPoint &target)
{
  return sqrt(pow(source[0] - target.x, 2) + pow(source[1] - target.y, 2));
}


unsigned int HeuristicChebishev(const PriorityPoint &source, const PriorityPoint &target)
{
  unsigned int a = abs(source.x - target.x);
  unsigned int b = abs(source.y - target.y);
  return a > b ? a  : b ;
}
unsigned int HeuristicManhetten(const PriorityPoint &source, const PriorityPoint &target)
{
  return (abs(source.x - target.x) + abs(source.y - target.y));
}
unsigned int HeuristicManhetten(int* source, const PriorityPoint &target)
{
  return (abs(source[0] - target.x) + abs(source[1] - target.y));
}