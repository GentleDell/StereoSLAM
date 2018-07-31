#ifndef USRMATH_H
#define USRMATH_H

#include <stdio.h>
#include <vector>
#include <iostream>
#include <numeric>

#include <opencv2/core.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/surface_matching.hpp"

using namespace std;

class Usrmath
{
public:

/// Estimate R & t
    cv::Matx44d matched_ICP(std::vector< cv::Point3f > vsrcPointCloud,
                        std::vector< cv::Point3f > vdstPointCloud, int flag = 0);

    cv::Matx44d standard_ICP(cv::Mat srcPC, cv::Mat dstPC, cv::Matx44d initialpose);

    cv::Matx44d standard_PnP(std::vector< cv::Point3f > vPointcloud,
                             std::vector< cv::Point2f > vimagePoints,
                             cv::Mat cameraMatrix, cv::Mat &pnpmask);

    cv::Mat decompose_essenMat();

};


#endif
