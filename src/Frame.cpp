#include"frame.h"


using namespace std;

const int EMPTYFRAME = -1;

// basic constructor
Frame::Frame()
{
    name = EMPTYFRAME;
}

// constructor generating necessary data
Frame::Frame(cv::Mat image1, cv::Mat image2, cv::Mat CamProjMat1, cv::Mat CamProjMat2,  Map &cloudmap, bool flag, int num)
{
    name = num;

    match_images(image1, image2);
    reprojectTo3D(CamProjMat1, CamProjMat2, cloudmap.cloudMap, flag);

    pframeTomap = &cloudmap;
}

void Frame::find_inliers(void)
{
    std::vector< cv::DMatch > good_matches;
    std::vector< cv::Point2f > vfeature_coordinate_l,vfeature_coordinate_r;
    cv::Mat flag_inliermatches;

    //-- Sort matches and preserve top 30% matches
    std::sort( vinframe_matches.begin(), vinframe_matches.end());  // in DMatch, "<" is defined to compare distance
    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(vinframe_matches.size() * GOOD_PORTION));  // ptsPairs <= GOOD_PTS_MAX
    for( int i = 0; i < ptsPairs; i++ )
    {
        good_matches.push_back( vinframe_matches[i] );
    }

    //-- obtain coordinates of feature points
    for( size_t i = 0; i < good_matches.size(); i++ )
   {
       vfeature_coordinate_l.push_back( vfeaturepoints_l[ good_matches[i].queryIdx ].pt );
       vfeature_coordinate_r.push_back( vfeaturepoints_r[ good_matches[i].trainIdx ].pt );
   }

    //-- estimate the fundamental matrix and refine positions of feature points
    /* After estimating the fundamental matrix, it would be better to conduct a guided search. */
    std::vector< cv::Point2f > newsrc_featpoint, newdst_featpoint;

    Frame::fundamentalMat = cv::findFundamentalMat(vfeature_coordinate_l, vfeature_coordinate_r, flag_inliermatches, cv::FM_RANSAC, 2, 0.99);

    cv::correctMatches(fundamentalMat, vfeature_coordinate_l, vfeature_coordinate_r, newsrc_featpoint, newdst_featpoint);

    //-- save inlier matches and refined features' coordinate
    for (size_t i = 0; i < good_matches.size(); i++){
        if (flag_inliermatches.at<int>(1,i)){
            vinframeinlier_matches.push_back(good_matches[i]);
            vfeaturepoints_l[ good_matches[i].queryIdx ].pt = newsrc_featpoint[i];
            vfeaturepoints_r[ good_matches[i].trainIdx ].pt = newdst_featpoint[i];
        }
    }
}

void Frame::match_images(cv::Mat image1, cv::Mat image2)
{

    // instantiate detectors/matchers
    SURFDetector surf;
    SURFMatcher<cv::BFMatcher> matcher;

    surf(image1, cv::Mat(), vfeaturepoints_l, descriptors_l);
    surf(image2, cv::Mat(), vfeaturepoints_r, descriptors_r);
    matcher.match(descriptors_l, descriptors_r, vinframe_matches);

    find_inliers();
}

void Frame::reprojectTo3D(cv::Mat ProjMat1, cv::Mat ProjMat2, std::vector< Mappoint > &v_mappoints, bool flag = 0)
{
    cv::Mat point4D;
    cv::Point3f triPoints;
    std::vector< cv::Point2f > vpoints1, vpoints2;

    if (flag){
        cv::Mat R1, R2, T1, T2; // Transformation matrics
        double x1, x2, y1, y2; // Image coordinates of give keypoints

        // ...
    }
    else{
        for (int i_ct = 0; i_ct < vinframeinlier_matches.size(); i_ct++ ){
            vpoints1.push_back( vfeaturepoints_l[ vinframeinlier_matches[i_ct].queryIdx ].pt );
            vpoints2.push_back( vfeaturepoints_r[ vinframeinlier_matches[i_ct].trainIdx ].pt );
        }

        cv::triangulatePoints(ProjMat1, ProjMat2, vpoints1, vpoints2, point4D);

        /* !Be careful that index starts from 0;
         * It might be better to save the point cloud in 4d and do not normalize */

        int maplength = v_mappoints.size();
        for(int i_ct = maplength; i_ct < maplength+point4D.cols; i_ct++){
            triPoints = cv::Point3f(point4D.at<float> (0, i_ct) / point4D.at<float> (3, i_ct),
                                    point4D.at<float> (1, i_ct) / point4D.at<float> (3, i_ct),
                                    point4D.at<float> (2, i_ct) / point4D.at<float> (3, i_ct));

            v_mappoints.push_back( Mappoint(triPoints, name, name, vinframeinlier_matches[i_ct]) );
            vMappoints_indexnum.push_back(i_ct);
        }
    }

    if(v_mappoints.empty())
    {
        cout << "ERROR: No mappoints are generated!" << endl;
        exit(EXIT_FAILURE);
    }
}

void Frame::checkframe()
{
    cout << "number of left feature:" << vfeaturepoints_l.size() << "\n"
         << "number of right feature:" << vfeaturepoints_r.size() << "\n"
         << "number of inlier matches:" << vinframeinlier_matches.size() << "\n"
         << "number of all matches:" << vinframe_matches.size() << "\n"
         << "number of mappoints:" << vMappoints_indexnum.size() << endl;
}

bool Frame::drawframe(cv::Mat image1, cv::Mat image2, int drawing_mode, cv::Mat CamProjMat = cv::Mat())
{
    if (drawing_mode == DRAW_LINE_BOTH)
    {
        cv::Mat img_matches;

        cv::drawMatches( image1, vfeaturepoints_l,
                     image2, vfeaturepoints_r,
                     vinframeinlier_matches, img_matches,
                     cv::Scalar(0,255,0), cv::Scalar(0,255,0),
                     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        cv::namedWindow("Matches", CV_WINDOW_NORMAL);
        cv::imshow("Matches",img_matches);
        cv::waitKey(10); // give sys enough time to draw the result

        return true;
    }
    else if(drawing_mode == DRAW_LINE_ONE)
    {
        cv::Mat newimage;
        cv::Point position(25, image1.rows-25);  // where the text will be written
        stringstream str;
        str << "Inlier matches:" << vinframeinlier_matches.size();

        //-- convert grayscale image to colorimage to make text clear
        if (image1.type() == CV_8UC1) {
          cv::cvtColor(image1, newimage, CV_GRAY2RGB);
        }

        //-- insert text
        cv::putText(newimage, str.str(), position, cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

        //-- draw cicles
        for (int i_ct = 0; i_ct < vinframeinlier_matches.size(); i_ct++){
            cv::line(newimage, vfeaturepoints_l[ vinframeinlier_matches[i_ct].queryIdx ].pt,
                    vfeaturepoints_r[ vinframeinlier_matches[i_ct].trainIdx ].pt, cv::Scalar(0,255,0));
        }

        cv::namedWindow("Matching result", cv::WINDOW_NORMAL);
        cv::imshow("Matching result", newimage);
        cv::waitKey(5);

        return true;

    }
    else if(drawing_mode == DRAW_POINT_ONE)
    {
        cv::Mat newimage;
        cv::Point position(25, image1.rows-25);  // where the text will be written
        stringstream str;
        str << "Inlier matches:" << vinframeinlier_matches.size();

        //-- convert grayscale image to colorimage to make text clear
        if (image1.type() == CV_8UC1) {
          cv::cvtColor(image1, newimage, CV_GRAY2RGB);
        }

        //-- insert text
        cv::putText(newimage, str.str(), position, cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 255), 1);

        //-- draw cicles
        for (int i_ct = 0; i_ct < vinframeinlier_matches.size(); i_ct++){
            cv::circle(newimage, vfeaturepoints_l[ vinframeinlier_matches[i_ct].queryIdx ].pt, 4, cv::Scalar(0, 255, 0));
        }

        cv::namedWindow("Matching result", cv::WINDOW_NORMAL);
        cv::imshow("Matching result", newimage);
        cv::waitKey(5);

        return true;
    }
    else if (drawing_mode == DRAW_3D_POINT)
    {
        if (CamProjMat.empty()){

            cout << "ERROR:" << endl;
            cout << "Project matrix of the right camera is not given!" << endl;

            return false;
        }

        bool camera_pov = 1;

        /// Create a window
        cv::viz::Viz3d myWindow("Coordinate Frame");

        /// Add coordinate axes
        myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

        /// Let's assume camera has the following properties
        cv::Vec3f cam_pos = cv::Vec3f(CamProjMat.col(3)) * -(1.0),
                  cam_focal_point = cv::Vec3f(CamProjMat.col(3))*(-1.0) - cv::Vec3f(0.0f,20.0f,0.0f),
                  cam_y_dir(0.0f,0.0f,-1.0f);

        /// We can get the pose of the cam using makeCameraPose
        cv::Affine3f cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

        /// We can get the transformation matrix from camera coordinate system to global using
        /// - makeTransformToGlobal. We need the axes of the camera
        cv::Affine3f transform = cv::viz::makeTransformToGlobal(cv::Vec3f(1.0f, 0.0f,0.0f), cv::Vec3f(0.0f, 1.0f, 0.0f),cv::Vec3f(0.0f,0.0f, 1.0f), cam_pos);

        /// Create a cloud widget.
        std::vector< cv::Point3f > pointcloud;
        for (int i_ct = 0; i_ct < pframeTomap->cloudMap.size(); i_ct++){
            pointcloud.push_back( pframeTomap->cloudMap[i_ct].position );
        }
        cv::viz::WCloud cloud_widget(pointcloud, cv::viz::Color::green());

        /// Pose of the widget in camera frame
        cv::Affine3f temp;
        cv::Affine3f cloud_pose = temp.translate(CamProjMat.col(3));
        /// Pose of the widget in global frame
        cv::Affine3f cloud_pose_global = transform * cloud_pose;

        /// Visualize widget
        myWindow.showWidget("features", cloud_widget, cloud_pose_global);

        /// Set the viewer pose to that of camera
        if (camera_pov)
            myWindow.setViewerPose(cam_pose);

        /// Start event loop.
        myWindow.spin();

        return true;
    }

}

