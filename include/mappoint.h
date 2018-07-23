#ifndef MAPPOINTS_H
#define MAPPOINTS_H

#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"

using namespace std;
using namespace cv;


class Mappoint
{
public:     // functions
    Mappoint();

    Mappoint(cv::Point3f point, int queryframename, int trainframename, cv::DMatch match);

public:     // variances

    cv::Point3f position;

    std::vector< int > v_visibleframes_query;    // vector of the name of Query frames containing this mappoint
    std::vector< int > v_visibleframes_train;    // vector of the name of Train frames containing this mappoint

    std::vector< cv::DMatch > v_visiblematches;   // vector of matches containing this mappoint
};


#endif
