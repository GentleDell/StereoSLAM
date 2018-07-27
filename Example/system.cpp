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

#include "map.h"
#include "frame.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const int KITTI_NAME_LENGTH = 6; // name length of images in KITTI dataset

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
    bool bUsemytriangular = 0; // 1: use my triangular algorithm, 0: use that of opencv
    bool funcflag = 1; // 0 use camera, 1 use KITTI 00;
    char * paramPathleft = "/home/zhant/Documents/camera2_left.yml";
    char * paramPathright = "/home/zhant/Documents/camera1_right.yml";

    Map globalMap;  // Globle point cloud Map

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

            for(int imgfile_ct = 0; ; imgfile_ct++)
            {
                capture_left >> left_image;
                capture_right >> right_image;

                if(left_image.empty() || right_image.empty())
                    break;

                left_image = calibrate_img(left_image, camera_Mat_left, distCoeffs_left);
                right_image = calibrate_img(right_image, camera_Mat_right, distCoeffs_right);

                Frame newframe = Frame(left_image, right_image, camera_Mat_left, camera_Mat_right, imgfile_ct);

                newframe.reprojectTo3D(globalMap.cloudMap, bUsemytriangular);

                newframe.drawframe(left_image, right_image, DRAW_3D_POINT);


                char key = (char)waitKey(5);
                if (key == 's'){
                    break;
                }

                imgfile_ct++;
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

        for(int imgfile_ct = 0; ; imgfile_ct ++)
        {
            if(!load_image(left_image, right_image, filepath, imgfile_ct))
            {
                return EXIT_FAILURE;
            }

            cout << "image pair:" << imgfile_ct << endl;

            Frame newframe = Frame(left_image, right_image, Pl, Pr, imgfile_ct);    // initialize frame 2D information
            newframe.reprojectTo3D(globalMap.cloudMap, bUsemytriangular);   // initialize 3D information
            newframe.pframeTomap = &globalMap;      // record global map in the frame
            globalMap.frameMap.push_back(newframe); // record frame in the global map

//            newframe.drawframe(left_image, right_image, DRAW_POINT_ONE);

            if (imgfile_ct >= 1){
                globalMap.frameMap[imgfile_ct -1].match_frames(globalMap.frameMap[imgfile_ct]);

                if (imgfile_ct%20 == 0)
                {
                    globalMap.draw_Map();
                }
            }

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
