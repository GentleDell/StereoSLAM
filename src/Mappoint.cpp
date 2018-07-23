#include "mappoint.h"

using namespace std;

//basic constructor
Mappoint::Mappoint()
{}

Mappoint::Mappoint(cv::Point3f point, int queryframename, int trainframename, cv::DMatch match)
{

    position = point;

    v_visibleframes_query.push_back(queryframename);
    v_visibleframes_train.push_back(trainframename);

    v_visiblematches.push_back(match);
}

