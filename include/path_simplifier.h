#ifndef PATH_SYMPLIFIER_H
#define PATH_SYMPLIFIER_H
#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <vector>

typedef std::pair<double, double> Point;

void FilterRosPath(nav_msgs::Path &simple, const nav_msgs::Path &full, const double &filter_epsilon);
void RamerDouglasPeucker(const std::vector<Point> &pointList, double epsilon, std::vector<Point> &out);
double PerpendicularDistance(const Point &pt, const Point &lineStart, const Point &lineEnd);

#endif
