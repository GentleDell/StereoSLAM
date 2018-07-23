#ifndef FRAME_H
#define FRAME_H

#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>

#include <opencv2/viz.hpp>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "map.h"
#include "mappoint.h"

using namespace std;
using namespace cv::xfeatures2d;

enum{
    /* MODE OF DRAW THIS FRAME */
    DRAW_LINE_BOTH      = 1,
    DRAW_LINE_ONE       = 2,
    DRAW_POINT_ONE      = 3,

    DRAW_3D_POINT       = 4
};

const int GOOD_PTS_MAX = 300;
const float GOOD_PORTION = 0.3f;


/* A frame includes avaliable information and operations of given stereo images.*/
class Frame
{
public:     // functions

    Frame();

    Frame(cv::Mat image1, cv::Mat image2, cv::Mat CamProjMat1, cv::Mat CamProjMat2, Map &cloudmap, bool flag, int num);

    void match_images(cv::Mat image1, cv::Mat image2);

    void find_inliers(void);

    /* REPROJECT 2D IMAGE FEATURE POINTS TO 3D MAP POINT.
     * Algrithm: An optimal solution mentioned in "Mutiple View Geometry in Computer Vision", Section 12.5(p315)
     *
     * Input arguments *
     * keypoints_left: vector of keypoints in the left image.
     * keypoints_right: vector of keypoints in the right image.
     * F : Fundamental Matrix.
     * flag : 1, using algrithm mentioned above; 0, using function of OpenCV.
     *        Default value is 1;
     *
     * Output argument *
     *
     */

    void reprojectTo3D(cv::Mat CamProjMat1, cv::Mat CamProjMat2, std::vector<Mappoint> &v_mappoints, bool flag);

    /* OUTPUT SAME VARIANCE OF THE CLASS OBJECT TO CHECK IT*/
    void checkframe();

    bool drawframe(cv::Mat image1, cv::Mat image2, int drawing_mode, cv::Mat CamProjMat);

    cv::Mat obtainrotation();
    cv::Mat obtaintranslation();


public:     // variances

    template<class KPMatcher>
    struct SURFMatcher
    {
        KPMatcher matcher;
        template<class T>
        void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
        {
            matcher.match(in1, in2, matches);
        }
    };

    struct SURFDetector
    {
        cv::Ptr<cv::Feature2D> surf;
        SURFDetector(double hessian = 800.0)
        {
            surf = SURF::create(hessian);
        }
        template<class T>
        void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
        {
            surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
        }
    };

    int name;

    cv::Mat T_w2c;  // pose of this frame

    std::vector< int > vMappoints_indexnum;   // Map points' number in Map

    std::vector< cv::DMatch > vinframe_matches, vinframeinlier_matches;   // matches of the given stereo images

    std::vector< vector< cv::DMatch > > voutframeinlier_matches;    // matches between this frame and other frames

    vector< int > voutframematches_number; // number of matched frames

    std::vector< cv::KeyPoint > vfeaturepoints_l, vfeaturepoints_r;     // featurepoints of left and right image

    cv::Mat descriptors_l, descriptors_r; // descriptors of above featurepoints

    cv::Mat fundamentalMat, essencialMat;

    Map *pframeTomap;   // pointer points to the global point cloud map

};


#endif
