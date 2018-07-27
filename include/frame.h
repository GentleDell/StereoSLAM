#ifndef FRAME_H
#define FRAME_H

#include <stdio.h>
#include <vector>
#include <iostream>
#include <sstream>

#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include <opencv2/viz/vizcore.hpp>      // need vtk
#include "opencv2/xfeatures2d.hpp"      // Opencv_contrib
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/surface_matching.hpp" // Opencv_contrib

#include "usrmath.h"
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

enum
{
    /* ALGORITHM FOR POSE ESTIMATION */
    PNP                 = 1,
    BPNP                = 2,
    ICP                 = 3,
    MICP                = 4,
    DEE                 = 5,
};

const int GOOD_PTS_MAX = 300;
const float GOOD_PORTION = 0.3f;

class Map;


/* A frame includes avaliable information and operations of given stereo images.*/
class Frame
{
public:

    Frame();

    Frame(cv::Mat image1, cv::Mat image2, cv::Mat projectmat1, cv::Mat projectmat2, int num);

///* INNER FRAME OPERATION
/// all these functions manipulate stereo image to form a frame
///
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
     *
     *
     */
    void reprojectTo3D( std::vector<Mappoint> &v_mappoints, bool flag );

    /* OUTPUT SAME VARIANCE OF THE CLASS OBJECT TO CHECK IT*/
    void checkframe();

    bool drawframe(cv::Mat image1, cv::Mat image2, int drawing_mode);

    cv::Mat obtainrotation();
    cv::Mat obtaintranslation();


///* INTER FRAME OPERATION
/// these functions work on frame level
///
    void match_frames(Frame &targetframe );

    void estimate_pose(std::vector< cv::DMatch > vinterframe_matchesinlier, Frame &targetframe , int algo);

    void find_inliers(std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2,
                      std::vector<DMatch>& matches, std::vector<DMatch>& inlier_matches);

    void reprojectInterFrameTo3D(Frame targetframe , std::vector<Mappoint> &v_mappoints );


public:

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

    cv::Matx44d T_w2c;  // pose of this frame

    cv::Mat CamProjMat_l, CamProjMat_r;     // Camera Project Matrix under camera coordinate of THIS FRAME

    std::vector< int > vMappoints_indexnum;   // Map points' number in global map

    /* Id of inframe triangulated points of left & right images */
    std::vector< int > vinframeinliermatches_queryIdx, vinframeinliermatches_trainIdx;

    /* Id of inter frame triangulated points of left & right images */
    std::vector< int > vinterframematch_queryIdx, vinterframematch_trainIdx;

    std::vector< cv::DMatch > vinframe_matches, vinframeinlier_matches;   // matches of the given stereo images

    std::vector< cv::KeyPoint > vfeaturepoints_l, vfeaturepoints_r;     // featurepoints of left and right image

    cv::Mat descriptors_l, descriptors_r; // descriptors of above featurepoints

    cv::Mat fundamentalMat, essencialMat;

    Map *pframeTomap;   // pointer points to the global point cloud map

};


#endif
