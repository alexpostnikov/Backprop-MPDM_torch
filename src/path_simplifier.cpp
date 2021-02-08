#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <vector>
#include <path_simplifier.h>

//Simplifies given path - remove useless points
void FilterRosPath(nav_msgs::Path &simple, const nav_msgs::Path &full, const double &filter_epsilon)
{
    Point tmp_point;
    std::vector<Point> vector_full;
    std::vector<Point> vector_simple;

    for (int i = 0; i < full.poses.size(); i++)
    {
        tmp_point = {full.poses[i].pose.position.x, full.poses[i].pose.position.y};
        vector_full.push_back(tmp_point);
    }
    RamerDouglasPeucker(vector_full, filter_epsilon, vector_simple);
    int out_size = vector_simple.size();
    simple.poses.clear();
    simple.header.stamp = full.header.stamp;
    simple.header.frame_id = full.header.frame_id;
    simple.poses.resize(out_size);
    for (int i = 0; i < out_size; i++)
    {
        simple.poses[i].pose.position.x = vector_simple[i].first;
        simple.poses[i].pose.position.y = vector_simple[i].second;
        simple.poses[i].pose.orientation = full.poses[0].pose.orientation; //TODO: simplify orientation
        simple.poses[i].header.stamp = full.header.stamp;
        simple.poses[i].header.frame_id = full.header.frame_id;
    }
}

void RamerDouglasPeucker(const std::vector<Point> &pointList, double epsilon, std::vector<Point> &out)
{
    if (pointList.size() < 2)
    {
        // ROS_ERROR("Not enough points to simplify path");
        return;
    }

    // Find the point with the maximum distance from line between start and end
    double dmax = 0.0;
    size_t index = 0;
    size_t end = pointList.size() - 1;
    for (size_t i = 1; i < end; i++)
    {
        double d = PerpendicularDistance(pointList[i], pointList[0], pointList[end]);
        if (d > dmax)
        {
            index = i;
            dmax = d;
        }
    }

    // If max distance is greater than epsilon, recursively simplify
    if (dmax > epsilon)
    {
        // Recursive call
        std::vector<Point> recResults1;
        std::vector<Point> recResults2;
        std::vector<Point> firstLine(pointList.begin(), pointList.begin() + index + 1);
        std::vector<Point> lastLine(pointList.begin() + index, pointList.end());
        RamerDouglasPeucker(firstLine, epsilon, recResults1);
        RamerDouglasPeucker(lastLine, epsilon, recResults2);

        // Build the result list
        out.assign(recResults1.begin(), recResults1.end() - 1);
        out.insert(out.end(), recResults2.begin(), recResults2.end());
        if (out.size() < 2)
        {
            ROS_ERROR("Problem assembling output");
            return;
        }
    }
    else
    {
        //Just return start and end points
        out.clear();
        out.push_back(pointList[0]);
        out.push_back(pointList[end]);
    }
}

double PerpendicularDistance(const Point &pt, const Point &lineStart, const Point &lineEnd)
{
    double dx = lineEnd.first - lineStart.first;
    double dy = lineEnd.second - lineStart.second;

    //Normalise
    double mag = pow(pow(dx, 2.0) + pow(dy, 2.0), 0.5);
    if (mag > 0.0)
    {
        dx /= mag;
        dy /= mag;
    }

    double pvx = pt.first - lineStart.first;
    double pvy = pt.second - lineStart.second;

    //Get dot product (project pv onto normalized direction)
    double pvdot = dx * pvx + dy * pvy;

    //Scale line direction vector
    double dsx = pvdot * dx;
    double dsy = pvdot * dy;

    //Subtract this from pv
    double ax = pvx - dsx;
    double ay = pvy - dsy;

    return pow(pow(ax, 2.0) + pow(ay, 2.0), 0.5);
}
