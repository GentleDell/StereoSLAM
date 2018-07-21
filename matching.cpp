#include <iostream>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <gsl/gsl_poly.h>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const int KITTI_NAME_LENGTH = 6; // name length of images in KITTI dataset
const int GOOD_PTS_MAX = 200;
const float GOOD_PORTION = 0.3f;

//void match_img(cv::Mat & left_image, cv::Mat & righ_image);
//void mix_image(cv::Mat & left_image, cv::Mat & righ_image);
//void read_param(cv::Mat & camera_Mat, cv::Mat & distCoeffs, cv::Size imageSize, char * IntrinsicsPath);
//Mat calibrate_img(cv::Mat & image, cv::Mat & camera_Mat, cv::Mat & distCoeffs);
//void find_polynomialroots (double coeff[], double roots[], int size);
// bool load_image(cv::Mat & left_image, cv::Mat & right_image, std::string filepath, int imgfile_ct);

struct SURFDetector
{
    Ptr<Feature2D> surf;
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

void mix_image(Mat & left_image, Mat & righ_image)
{
    int rows = left_image.rows>righ_image.rows?left_image.rows:righ_image.rows;//合成图像的行数
    int cols = left_image.cols+10+righ_image.cols;  //合成图像的列数
    int gap = 0;
    Mat mixed_image = Mat::zeros(rows, cols, left_image.type());

    CV_Assert(left_image.type()==righ_image.type());

    left_image.copyTo(mixed_image(cv::Rect(0,0,left_image.cols,left_image.rows)));
    righ_image.copyTo(mixed_image(cv::Rect(left_image.cols+gap,0,righ_image.cols,righ_image.rows)));//两张图像之间相隔gap个像素
    cv::imshow("the two images", mixed_image);
}

void read_param(Mat & camera_Mat, Mat & distCoeffs, Size imageSize, char * IntrinsicsPath)
{
       bool FSflag = false;
       FileStorage readfs;

       FSflag = readfs.open(IntrinsicsPath, FileStorage::READ);
       if (FSflag == false){
           cout << "Cannot open camera_param file" << endl;
           exit(0);
       }

       readfs["camera_matrix"] >> camera_Mat;
       readfs["distortion_coefficients"] >> distCoeffs;
       readfs["image_width"] >> imageSize.width;
       readfs["image_height"] >> imageSize.height;

       cout << camera_Mat << endl << distCoeffs << endl << imageSize << endl;

       readfs.release();
}

Mat calibrate_img(Mat & image, Mat & camera_Mat, Mat & distCoeffs)
{
    Mat undist_img;

    undistort(image, undist_img, camera_Mat, distCoeffs);

    return undist_img;
}

bool load_cameraparam(cv::Mat Pl, cv::Mat Pr, char* paramfilepath)
{
    int ct_i, calib_ct = 12;
    float temp_float;
    string calibparam, temp_string;
    std::ifstream infile(paramfilepath);
    stringstream temp_ss;

    if (!infile.is_open()){
        cout << "fail to open calib.txt" << endl;
        return false;
    }
    else{
        while(getline(infile,temp_string)){
            temp_ss << temp_string;
            temp_ss >> calibparam;
            if(calibparam == "P0:"){
                for (ct_i = 0; ct_i < calib_ct; ct_i++){
                    temp_ss >> temp_float;
                    Pl.at<float>(ct_i/4, ct_i%4) = temp_float;
                }
                temp_ss.clear();
                temp_ss.str("");
            }
            else if(calibparam == "P1:"){
                for (int ct_i = 0; ct_i < calib_ct; ct_i++){
                    temp_ss >> temp_float;
                    Pr.at<float>(ct_i/4, ct_i%4) = temp_float;
                }
                temp_ss.clear();
                temp_ss.str("");
                break;      // only read P0 and P1
            }

        }
        infile.close();

        return true;
    }
}

bool load_image(cv::Mat & left_image, cv::Mat & right_image, std::string filepath, int imgfile_ct)
{
    int length_ss = 0;
    stringstream temp_ss; // to obtain calib data and generate the file name;
    std::string leftimagepath, rightimagepath;

    temp_ss << imgfile_ct;
    length_ss = temp_ss.str().size();
    temp_ss.str("");

    while(length_ss < KITTI_NAME_LENGTH){
        temp_ss << 0;
        length_ss++;
    }
    temp_ss << imgfile_ct;
    leftimagepath = filepath + "image_0/" + temp_ss.str() + ".png";
    rightimagepath = filepath + "image_1/" + temp_ss.str() + ".png";

    imread(leftimagepath, IMREAD_GRAYSCALE).copyTo(left_image);
    if(left_image.empty())
    {
        std::cout << "Couldn't load image in" << leftimagepath << std::endl;
        return false;
    }

    imread(rightimagepath, IMREAD_GRAYSCALE).copyTo(right_image);
    if(right_image.empty())
    {
        std::cout << "Couldn't load image in" << rightimagepath << std::endl;
        return false;
    }

    return true;
}

void find_inliers(std::vector<KeyPoint>& keypoints1,
                  std::vector<KeyPoint>& keypoints2,
                  std::vector<DMatch>& matches,
                  std::vector<DMatch>& inlier_matches)
{
    std::vector< DMatch > good_matches;
    std::vector<Point2f> src_featpoint, dst_featpoint;
    cv::Mat fundamentalMat, flag_inliermatches;

    //-- Sort matches and preserve top 30% matches
    std::sort(matches.begin(), matches.end());  // in DMatch, "<" is defined to compare distance
    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(matches.size() * GOOD_PORTION));  // ptsPairs <= GOOD_PTS_MAX
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( matches[i] );
    }

    //-- obtain coordinates of feature points
    for( size_t i = 0; i < good_matches.size(); i++ )
   {
       src_featpoint.push_back( keypoints1[ good_matches[i].queryIdx ].pt );  //??? need to del useless keypoints?
       dst_featpoint.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
   }

    //-- estimate the fundamental matrix and refine positions of feature points
    std::vector<Point2f> newsrc_featpoint, newdst_featpoint;
    fundamentalMat = findFundamentalMat(src_featpoint, dst_featpoint, flag_inliermatches, FM_RANSAC, 2, 0.99);
    correctMatches(fundamentalMat, src_featpoint, dst_featpoint, newsrc_featpoint, newdst_featpoint);

    /* After estimating fundamatal matrix, it would be better to conduct a guided search for matches */


    for (size_t i = 0; i < good_matches.size(); i++){
        if (flag_inliermatches.at<int>(1,i)){
            inlier_matches.push_back(good_matches[i]);
            keypoints1[ good_matches[i].queryIdx ].pt = newsrc_featpoint[i];
            keypoints2[ good_matches[i].trainIdx ].pt = newdst_featpoint[i];
        }
    }
}

std::vector<DMatch> match_img(cv::Mat& left_image, cv::Mat& righ_image,
                              std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2,
                              cv::Mat& descriptors1, cv::Mat& descriptors2)
{
    std::vector<DMatch> matches, inlier_matches;

    // instantiate detectors/matchers
    SURFDetector surf;
    SURFMatcher<BFMatcher> matcher;

    surf(left_image, Mat(), keypoints1, descriptors1);
    surf(righ_image, Mat(), keypoints2, descriptors2);
    matcher.match(descriptors1, descriptors2, matches);

    find_inliers(keypoints1, keypoints2, matches, inlier_matches);

    return inlier_matches;
}

/* ESTIMATE FUNDAMENTAL MATRICS, version 0.01, RANSAC is not included.
 * Algorithm: normalized 8-point algoritm, in "Mutiple View Geometry in Computer Vision", Section 11.2 (p282)
 *            Iterative algorithm could be involved in later version.
 */
cv::Mat find_FundaMat(std::vector<KeyPoint> keypoints_left, std::vector<KeyPoint> keypoints_right, std::vector<DMatch> inlier_matches){

    int point_count = inlier_matches.size();
    vector<Point2f> points1(point_count);
    vector<Point2f> points2(point_count);

    // initialize the points here ...
    for( int i = 0; i < point_count; i++ )
    {
        points1[i] = keypoints_left[inlier_matches[i].queryIdx].pt;
        points2[i] = keypoints_right[inlier_matches[i].trainIdx].pt;
    }

    Mat fundamental_matrix =
     findFundamentalMat(points1, points2, FM_RANSAC, 2, 0.99);

    return fundamental_matrix;
}

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
std::vector< Point3f > reprojectTo3D(
        std::vector<KeyPoint> vfeaturepoint1,
        std::vector<KeyPoint> vfeaturepoint2,
        std::vector< cv::DMatch > matches,
        cv::Mat ProjMat1 = cv::Mat(),      // knowledge: Mat& can not be initiated as a Mat as it is a address
        cv::Mat ProjMat2 = cv::Mat(),
        cv::Mat FundaMat = cv::Mat(),
        bool flag = 0)
{
    int i_ct;
    std::vector< Point3f > vtriangularpoints;

    if (flag){
        cv::Mat R1, R2, T1, T2; // Transformation matrics
        double x1, x2, y1, y2; // Image coordinates of give keypoints

        // ...

    }
    else{
        cv::Mat point4D;
        cv::Point3f triPoints;
        std::vector< Point2f > vpoints1, vpoints2;

        for ( i_ct = 0; i_ct < matches.size(); i_ct++ ){
            vpoints1.push_back( vfeaturepoint1[ matches[i_ct].queryIdx ].pt );
            vpoints2.push_back( vfeaturepoint2[ matches[i_ct].trainIdx ].pt );
        }

        cv::triangulatePoints(ProjMat1, ProjMat2, vpoints1, vpoints2, point4D);

        /* !Be careful that index starts from 0;
         * It might be better to save the point cloud in 4d and do not normalize */
        for(i_ct = 0; i_ct < point4D.cols; i_ct++){
            triPoints.x = point4D.at<float> (0, i_ct) / point4D.at<float> (3, i_ct);
            triPoints.y = point4D.at<float> (1, i_ct) / point4D.at<float> (3, i_ct);
            triPoints.z = point4D.at<float> (2, i_ct) / point4D.at<float> (3, i_ct);
            vtriangularpoints.push_back(triPoints);
        }
    }
    return vtriangularpoints;
}


void find_polynomialroots (double coeff[], double roots[], int size)
{
    int ct_i;

    gsl_poly_complex_workspace * workspace = gsl_poly_complex_workspace_alloc (size);
    gsl_poly_complex_solve (coeff, size, workspace, roots);
    gsl_poly_complex_workspace_free (workspace);

    for (ct_i = 0; ct_i < size-1; ct_i++)
    {
        printf ("z%d = %+.18f %+.18f\n", ct_i, roots[2*ct_i], roots[2*ct_i+1]);
    }
}

int main()
{
    cv::Mat left_image, right_image;

    cv::Mat descriptors1, descriptors2; // cv::UMat is a new container in opencv3 for gpu-based image operating
    std::vector< KeyPoint > keypoints1, keypoints2;
    std::vector< Point3f > v3DPoints;
    std::vector< DMatch > inlier_matches;

    bool funcflag = 1, drawflag = 1; // 0 use camera, 1 use KITTI 00;
    char * paramPathleft = "/home/zhant/Documents/camera2_left.yml";
    char * paramPathright = "/home/zhant/Documents/camera1_right.yml";

    cout << "Built with OpenCV " << CV_VERSION << endl;

    /* using realtime video
       obtaining images from usb camera*/
    if (!funcflag){

        cv::Size imageSize;
        cv::VideoCapture capture_left,  capture_right;
        cv::Mat camera_Mat_left, camera_Mat_right,  // 3*3 Mat
                distCoeffs_left, distCoeffs_right;

        cout << "reading undistortion parameters" << endl;
        read_param(camera_Mat_left, distCoeffs_left, imageSize, paramPathleft);
        read_param(camera_Mat_right, distCoeffs_right, imageSize, paramPathright);
        capture_left.open("/dev/leftcam");
        capture_right.open("/dev/rightcam");
        if(capture_left.isOpened() && capture_right.isOpened()) {
            cout << "Both are opened" << endl;
            for(;;){
                capture_left >> left_image;
                capture_right >> right_image;

                if(left_image.empty() || right_image.empty())
                    break;

                left_image = calibrate_img(left_image, camera_Mat_left, distCoeffs_left);
                right_image = calibrate_img(right_image, camera_Mat_right, distCoeffs_right);

                inlier_matches = match_img(left_image, right_image, keypoints1, keypoints2, descriptors1, descriptors2);

                char key = (char)waitKey(5);
                if (key == 's'){
                    break;
                }
            }
        }
        else{
            cout << "One or more camera are failed" << endl;
            waitKey(0);
        }
    }

    /* read KITTI 00 image sequence
                                */
    else{
        char * paramfilepath = "/home/zhant/Documents/00/calib.txt";
        std::string filepath = "/home/zhant/Documents/00/";
        cv::Mat Pl(3,4,CV_32F), Pr(3,4,CV_32F);     // Projection matrix of left and right cameras

        if(!load_cameraparam(Pl,Pr,paramfilepath))
        {
            return EXIT_FAILURE;
        }
        cout << Pl << "\n" << Pr << endl;

        int imgfile_ct = 0;
        for(;;){

            if(!load_image(left_image, right_image, filepath, imgfile_ct)){
                return EXIT_FAILURE;
            }

            inlier_matches = match_img(left_image, right_image, keypoints1, keypoints2,
                                       descriptors1, descriptors2);

            v3DPoints = reprojectTo3D(keypoints1, keypoints2, inlier_matches, Pl, Pr);


//            if (drawflag){
//                Mat img_matches;
//                drawMatches( left_image, keypoints1, right_image, keypoints2, inlier_matches, img_matches, Scalar(0,255,0), Scalar(0,255,0),  // setting color
//                             std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );
//                cv::namedWindow("Matches", CV_WINDOW_NORMAL);
//                cv::imshow("Matches",img_matches);
//                waitKey(0); // give sys enough time to draw the result
//            }

            imgfile_ct++;
        }
    }
    return EXIT_SUCCESS;
}


//// call the roots finding
//int main(void){
//    int size = 6;
//    double a[6] = { -1, 0, 0, 0, 0, 1 };
//    double roots[10];
//    find_polynomialroots(a,roots,size);

//    return EXIT_SUCCESS;
//}


//// calibrated images
// left_image = calibrate_img(left_image, camera_Mat_left, distCoeffs_left);
// right_image = calibrate_img(right_image, camera_Mat_right, distCoeffs_right);

//// show matches
//if (drawflag){
//    Mat img_matches;
//    drawMatches( left_image, keypoints1, right_image, keypoints2, inlier_matches, img_matches, Scalar(0,255,0), Scalar(0,255,0),  // setting color
//                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );
//    cv::namedWindow("Matches", CV_WINDOW_NORMAL);
//    cv::imshow("Matches",img_matches);
//    waitKey(0); // give sys enough time to draw the result
