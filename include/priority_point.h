#ifndef PRIORITY_POINT_H
#define PRIORITY_POINT_H

#include <ros/ros.h>
#include <cmath>

class PriorityPoint
{
public:
  double priority;
  int x;
  int y;
  double cell_cost;
  PriorityPoint()
  {
    x = 0;
    y = 0;
    priority = 0;
    cell_cost = 0;
  }
  PriorityPoint(int x_, int y_)
  {
    x = x_;
    y = y_;
    priority = 0;
  }
  PriorityPoint(int x_, int y_, double priority_)
  {
    x = x_;
    y = y_;
    priority = priority_;
  }
  PriorityPoint(int x_, int y_, double priority_, double cell_cost_)
  {
    x = x_;
    y = y_;
    priority = priority_;
    cell_cost = cell_cost_;
  }
  bool operator>(const PriorityPoint &p2) const
  {
    return !(priority > p2.priority);
  }
  bool operator<(const PriorityPoint &p2) const
  {
    return !(priority < p2.priority);
  }
  bool operator==(const PriorityPoint &p2) const
  {
    return (x == p2.x) && (y == p2.y);
  }
  bool operator!=(const PriorityPoint &p2) const
  {
    return (x != p2.x) || (y != p2.y);
  }
  void operator=(const PriorityPoint &p2)
  {
    x = p2.x;
    y = p2.y;
    priority = p2.priority;
    cell_cost = p2.cell_cost;
  }
  void operator=(const int *p2)
  {
    x = p2[0];
    y = p2[1];
  }
  void GetNeighbors(PriorityPoint *neighbors)
  {
    neighbors[0] = PriorityPoint(x, y - 1, 0.0);
    neighbors[1] = PriorityPoint(x, y + 1, 0.0);
    neighbors[2] = PriorityPoint(x - 1, y, 0.0);
    neighbors[3] = PriorityPoint(x + 1, y, 0.0);
    double surcharge = sqrt(2) * cell_cost - cell_cost;
    neighbors[4] = PriorityPoint(x - 1, y - 1, surcharge);
    neighbors[5] = PriorityPoint(x - 1, y + 1, surcharge);
    neighbors[6] = PriorityPoint(x + 1, y - 1, surcharge);
    neighbors[7] = PriorityPoint(x + 1, y + 1, surcharge);
  }
  void GetNeighbors(PriorityPoint *neighbors,unsigned int dist)
  {
    neighbors[0] = PriorityPoint(x, y - dist, 0.0);
    neighbors[1] = PriorityPoint(x, y + dist, 0.0);
    neighbors[2] = PriorityPoint(x - dist, y, 0.0);
    neighbors[3] = PriorityPoint(x + dist, y, 0.0);
    double surcharge = (sqrt(2) * cell_cost - cell_cost)*dist; 
    neighbors[4] = PriorityPoint(x - dist, y - dist, surcharge);
    neighbors[5] = PriorityPoint(x - dist, y + dist, surcharge);
    neighbors[6] = PriorityPoint(x + dist, y - dist, surcharge);
    neighbors[7] = PriorityPoint(x + dist, y + dist, surcharge);
  }

  inline bool OnMap(const size_t max_x, const size_t max_y)
  {
    return (x > 0) && (x < max_x) && (y > 0) && (y < max_y) ? true : false;
  }
  
};

struct myCompare
{
  bool operator()(const PriorityPoint &p1, const PriorityPoint &p2)
  {
    return p1.priority > p2.priority;
  }
};
#endif