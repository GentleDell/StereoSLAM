#include "map.h"
#include "frame.h"

using namespace std;

const int EMPTY = -1;

// basic constructor
Frame::Frame()
{
    name = EMPTY;
}

// constructor generating necessary data
Frame::Frame(cv::Mat image1, cv::Mat image2, cv::Mat projectmat1, cv::Mat projectmat2, int num)
{
    name = num;
    T_w2c = Matx44d::eye();
    pframeTomap = NULL;
    CamProjMat_l = projectmat1;
    CamProjMat_r = projectmat2;

    match_images(image1, image2);
//    reprojectTo3D(CamProjMat1, CamProjMat2, flag);

}


///* Funcitons below manipulate stereo image to form a frame
///
void Frame::find_inliers(void)
{
    std::vector< cv::DMatch > good_matches;
    std::vector< cv::Point2f > vfeature_coordinate_l,vfeature_coordinate_r;
    cv::Mat flag_inliermatches;

    /* Sort matches and preserve top GOOD_PORTION% of filted matches.
     * Filting matches here is better, since enough points are provided
     * for triangulation and interframe matching */
    int matches_ct = 0;
    int max_ct = vinframe_matches.size();
    std::sort( vinframe_matches.begin(), vinframe_matches.end());  // in DMatch, "<" is defined to compare distance
    const int ptsPairs = std::min(GOOD_PTS_MAX, (int)(vinframe_matches.size() * GOOD_PORTION));  // ptsPairs <= GOOD_PTS_MAX
    for( int i = 0; i < ptsPairs && matches_ct < max_ct ; i++ )
    {
       good_matches.push_back( vinframe_matches[matches_ct] );
       matches_ct++;

       vfeature_coordinate_l.push_back( vfeaturepoints_l[ good_matches[i].queryIdx ].pt );
       vfeature_coordinate_r.push_back( vfeaturepoints_r[ good_matches[i].trainIdx ].pt );

       /* Discard points behind the camera */
       if(vfeature_coordinate_l[i].x <= vfeature_coordinate_r[i].x)
       {
           good_matches.pop_back();
           vfeature_coordinate_l.pop_back();
           vfeature_coordinate_r.pop_back();
           i--;
           continue;
       }

       /* If camera is vertically rectified, points with too much slip will be discarded */
       if( VERTICAL_REC && (abs(vfeature_coordinate_l[i].y - vfeature_coordinate_r[i].y) > MAX_VERTICAL_DISPARITY) )
       {
           good_matches.pop_back();
           vfeature_coordinate_l.pop_back();
           vfeature_coordinate_r.pop_back();
           i--;
           continue;
       }

       /* Discard distant points to reduce effects of error */
       if ( (vfeature_coordinate_l[i].x - vfeature_coordinate_r[i].x)*(vfeature_coordinate_l[i].x - vfeature_coordinate_r[i].x) +
            (vfeature_coordinate_l[i].y - vfeature_coordinate_r[i].y)*(vfeature_coordinate_l[i].y - vfeature_coordinate_r[i].y)
             < MIN_DISPARITY*MIN_DISPARITY )
       {
           good_matches.pop_back();
           vfeature_coordinate_l.pop_back();
           vfeature_coordinate_r.pop_back();
           i--;
           continue;
       }
    }

    //-- estimate the fundamental matrix and refine positions of feature points
    /* After estimating the fundamental matrix, it would be better to conduct a guided search. */
    std::vector< cv::Point2f > newsrc_featpoint, newdst_featpoint;

    fundamentalMat = cv::findFundamentalMat(vfeature_coordinate_l, vfeature_coordinate_r, flag_inliermatches, cv::FM_RANSAC, 2, 0.99);

    cv::correctMatches(fundamentalMat, vfeature_coordinate_l, vfeature_coordinate_r, newsrc_featpoint, newdst_featpoint);

    //-- save inlier matches and refined features' coordinate
    for (size_t i = 0; i < good_matches.size(); i++){
        if (flag_inliermatches.at<int>(1,i))
        {
            vinframeinlier_matches.push_back(good_matches[i]);
            vfeaturepoints_l[ good_matches[i].queryIdx ].pt = newsrc_featpoint[i];
            vfeaturepoints_r[ good_matches[i].trainIdx ].pt = newdst_featpoint[i];

            vinframeinliermatches_queryIdx.push_back( good_matches[i].queryIdx );
            vinframeinliermatches_trainIdx.push_back( good_matches[i].trainIdx );
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

void Frame::reprojectTo3D(std::vector< Mappoint > &v_mappoints, bool flag = 0)
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
            vpoints1.push_back( vfeaturepoints_l[ vinframeinliermatches_queryIdx[i_ct] ].pt );
            vpoints2.push_back( vfeaturepoints_r[ vinframeinliermatches_trainIdx[i_ct] ].pt );
        }

        cv::triangulatePoints(CamProjMat_l, CamProjMat_r, vpoints1, vpoints2, point4D);

        /* !Be careful that index starts from 0;
         * It might be better to save the point cloud in 4d and do not normalize */

        int maplength = v_mappoints.size();
        for(int i_ct = 0; i_ct < point4D.cols; i_ct++){
            triPoints = cv::Point3f(point4D.at<float> (0, i_ct) / point4D.at<float> (3, i_ct),
                                    point4D.at<float> (1, i_ct) / point4D.at<float> (3, i_ct),
                                    point4D.at<float> (2, i_ct) / point4D.at<float> (3, i_ct));

            v_mappoints.push_back( Mappoint(triPoints, name, name, vinframeinlier_matches[i_ct]) );
            vMappoints_indexnum.push_back(i_ct + maplength);
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

bool Frame::drawframe(cv::Mat image1, cv::Mat image2, int drawing_mode)
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
            cv::line(newimage, vfeaturepoints_l[ vinframeinliermatches_queryIdx[i_ct] ].pt,
                    vfeaturepoints_r[ vinframeinliermatches_trainIdx[i_ct] ].pt, cv::Scalar(0,255,0));
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
            cv::circle(newimage, vfeaturepoints_l[ vinframeinliermatches_queryIdx[i_ct] ].pt, 4, cv::Scalar(0, 255, 0));
        }

        cv::namedWindow("Matching result", cv::WINDOW_NORMAL);
        cv::imshow("Matching result", newimage);
        cv::waitKey(5);

        return true;
    }
    else if (drawing_mode == DRAW_3D_POINT)
    {
        if (pframeTomap == NULL){
            cout << "ERROR: " << "\n"
                 << "the conection between this frame and global map is not constructed now" << endl;

            return false;
        }

        bool camera_pov = 1;

        // Create a window
        cv::viz::Viz3d myWindow("Coordinate Frame");

        // Add coordinate axes
        myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

        // Let's assume camera has the following properties
        cv::Vec3f cam_pos = cv::Vec3f(CamProjMat_r.col(3)) * -(1.0),
                  cam_focal_point = cv::Vec3f(CamProjMat_r.col(3))*(-1.0) - cv::Vec3f(0.0f,20.0f,0.0f),
                  cam_y_dir(0.0f,0.0f,-1.0f);

        // We can get the pose of the cam using makeCameraPose
        cv::Affine3f cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

        // We can get the transformation matrix from camera coordinate system to global using
        // - makeTransformToGlobal. We need the axes of the camera
        cv::Affine3f transform = cv::viz::makeTransformToGlobal(cv::Vec3f(1.0f, 0.0f,0.0f), cv::Vec3f(0.0f, 1.0f, 0.0f),cv::Vec3f(0.0f,0.0f, 1.0f), cam_pos);

        // Create a cloud widget.
        std::vector< cv::Point3f > pointcloud;
        for (int i_ct = 0; i_ct < vMappoints_indexnum.size(); i_ct++){
            pointcloud.push_back( pframeTomap->cloudMap[ vMappoints_indexnum[i_ct] ].position );
        }
        cv::viz::WCloud cloud_widget(pointcloud, cv::viz::Color::green());

        // Pose of the widget in camera frame
        cv::Affine3f temp;
        cv::Affine3f cloud_pose = temp.translate(CamProjMat_r.col(3));
        /// Pose of the widget in global frame
        cv::Affine3f cloud_pose_global = transform * cloud_pose;

        // Visualize widget
        myWindow.showWidget("features", cloud_widget, cloud_pose_global);

        // Set the viewer pose to that of camera
        if (camera_pov)
            myWindow.setViewerPose(cam_pose);

        // Start event loop.
        myWindow.spin();

        return true;
    }

}


///* Funcitons below implement on frame level, generating interframe matches and 3D map points
///
void Frame::find_inliers(std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2,
                         std::vector<DMatch>& matches, std::vector<DMatch>& inlier_matches)
{
    std::vector< DMatch > good_matches;
    std::vector<Point2f> src_featpoint, dst_featpoint;
    cv::Mat tempMat, flag_inliermatches;

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

    tempMat = findEssentialMat( src_featpoint, dst_featpoint, CamProjMat_l(Range(0,3), Range(0,3)),
                                     RANSAC, 0.999, 1.0, flag_inliermatches);

//    tempMat = cv::findFundamentalMat(src_featpoint, dst_featpoint, flag_inliermatches, cv::FM_RANSAC, 2, 0.99);


    /* After estimating fundamatal matrix, it would be better to conduct a guided search for matches */

    for (size_t i = 0; i < good_matches.size(); i++){
        if (flag_inliermatches.at<int>(1,i)){
            inlier_matches.push_back(good_matches[i]);
        }
    }
}

/* structure to record matches and corresponding 3D points */
struct Match_3D_corresp
{
    Match_3D_corresp() {}

    cv::DMatch match;

    int index_3D_frame1;    // the index of 3D point of frame1

    int index_3D_frame2;    // the index of 3D point of frame2
};

/* DIVID INLIERS INTO 3 GROUPS & ESTIMATE POSE:
 *   vmatches_bothnew: Matches in this group are tatolly new. After estmating R&t of target frame,
 * all matches in this group will be triangulate to 3D map point could.
 *
 *   vmatches_1stnew & vmatches_2ndnew: in this group, all these matches have a new feature point
 * that has not been triangulated yet. The other point of the matches has been mapped. As aresult,
 * we can us PnP on matches of this group to estimate R&t.
 *
 *   vmatches_nonew: Points forming these matches have already been triangulated. Therefore, we are
 * able conduct ICP algorithm on matches of this group to estimate R&t.
 *
 * In fact, fusing all these estimation should be better, but in the present, we aim to accomplish
 * the whole system. we would like to accompish fusion and optimization later.
 * */
 void Frame::estimate_pose( std::vector< cv::DMatch > vinterframe_matchesinlier, Frame &targetframe, int algo )
{
    int distance;
    Match_3D_corresp temp;
    std::vector< Match_3D_corresp > vcorresp_bothnew, vcorresp_1stnew, vcorresp_2ndnew, vcorresp_nonew;

/// GROUPING POINT CLOUD
    for (int i_ct = 0; i_ct < vinterframe_matchesinlier.size(); i_ct++)
    {
        vector< int >::iterator this_result = find(vinframeinliermatches_queryIdx.begin(),
                                                   vinframeinliermatches_queryIdx.end(),
                                                   vinterframe_matchesinlier[i_ct].queryIdx);

        vector< int >::iterator target_result = find(targetframe.vinframeinliermatches_queryIdx.begin(),
                                                     targetframe.vinframeinliermatches_queryIdx.end(),
                                                     vinterframe_matchesinlier[i_ct].trainIdx);

        temp.match = vinterframe_matchesinlier[i_ct];
        temp.index_3D_frame1 = EMPTY;
        temp.index_3D_frame2 = EMPTY;

        // a "new" interframe matches in both frames
        if( (this_result == vinframeinliermatches_queryIdx.end())
                && (target_result == targetframe.vinframeinliermatches_queryIdx.end()) )
        {
            vcorresp_bothnew.push_back( temp );
        }
        // the target-frame point has matched with a point in right image of target frame
        else if( (this_result == vinframeinliermatches_queryIdx.end())
                && (target_result != targetframe.vinframeinliermatches_queryIdx.end()) )
        {
            distance = std::distance(targetframe.vinframeinliermatches_queryIdx.begin(), target_result);
            temp.index_3D_frame2 = targetframe.vMappoints_indexnum[distance];

            vcorresp_1stnew.push_back( temp );
        }
        // a "new" interframe matches only in target frames
        else if( (this_result != vinframeinliermatches_queryIdx.end())
                 && (target_result == targetframe.vinframeinliermatches_queryIdx.end()) )
        {
            distance = std::distance(vinframeinliermatches_queryIdx.begin(), this_result);
            temp.index_3D_frame1 = vMappoints_indexnum[distance];

            vcorresp_2ndnew.push_back( temp );
        }
        // both point of the match has matched with other point in corresponding frame
        else
        {
            distance = std::distance(vinframeinliermatches_queryIdx.begin(), this_result);
            temp.index_3D_frame1 = vMappoints_indexnum[distance];

            distance = std::distance(targetframe.vinframeinliermatches_queryIdx.begin(), target_result);
            temp.index_3D_frame2 = targetframe.vMappoints_indexnum[distance];

            vcorresp_nonew.push_back( temp );
        }
    }

//    cout << "points for PNP:" << vcorresp_1stnew.size() << "\n"
//         << "points for BPNP::" << vcorresp_2ndnew.size() << "\n"
//         << "points for ICP & MICP:" << vcorresp_nonew.size() << "\n"
//         << "points for DEE:" << vcorresp_bothnew.size() << endl;

/// POSE ESTIMATION IS AT BELOW
/// preprocessing

    // when PNP is chosen, once points less than 4, BPNP would be used
    if ( vcorresp_1stnew.size() < 4 && algo == PNP )
    {
        cout << "PNP is chosen" << endl;
        algo = BPNP;
    }
    // when BPNP is chosen, once points less than 4
    if ( vcorresp_2ndnew.size() < 4 && algo == BPNP )
    {
        if (vcorresp_1stnew.size() < 4)
        {
            cout << "MICP is chosen" << endl;
            algo = MICP;
        }
        else
        {
            cout << "PNP is chosen" << endl;
            algo = PNP;
        }
    }

/// estiamting pose
///
    cv::Matx44d pose;
    Usrmath usr_operator;

    /* Standard ICP 3D-3D
     * Unstable with sparse point cloud even with a not bad initial pose. */
    if( algo == ICP )
    {
        int num_points = vcorresp_nonew.size();
        cv::Mat srcPointCloud(num_points, 3, CV_32F),
                dstPointCloud(num_points, 3, CV_32F);

        for (int i_ct = 0; i_ct < num_points; i_ct++)
        {
            float *row_src = srcPointCloud.ptr<float>(i_ct);

            row_src[0] = pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame1].position.x;
            row_src[1] = pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame1].position.y;
            row_src[2] = pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame1].position.z;

            float *row_dst = dstPointCloud.ptr<float>(i_ct);

            row_dst[0] = targetframe.pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame2].position.x;
            row_dst[1] = targetframe.pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame2].position.y;
            row_dst[2] = targetframe.pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame2].position.z;
        }

        pose = usr_operator.standard_ICP(srcPointCloud, dstPointCloud, targetframe.T_w2c);
    }

    /* Matched ICP 3D-3D */
    else if(algo == MICP)
    {
        std::vector< cv::Point3f > vsrcPointCloud;
        std::vector< cv::Point3f > vdstPointCloud;

        for (int i_ct = 0; i_ct < vcorresp_nonew.size(); i_ct++)
        {
            vsrcPointCloud.push_back( pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame1].position );
            vdstPointCloud.push_back( targetframe.pframeTomap->cloudMap[vcorresp_nonew[i_ct].index_3D_frame2].position );
        }

        pose = usr_operator.matched_ICP(vsrcPointCloud, vdstPointCloud);
    }

    /* Forward PnP 2D-3D
     * Not stable enough when there are a few points, mismatch is the main reason. */
    else if(algo == PNP)
    {
        std::vector< cv::Point3f > Pointcloud;
        std::vector< cv::Point2f > imagePoints;
        cv::Mat pnp_mask, cameraMatrix = CamProjMat_l(Range(0,3), Range(0,3));

        for (int i_ct = 0; i_ct < vcorresp_1stnew.size(); i_ct++)
        {
            Pointcloud.push_back( pframeTomap->cloudMap[ vcorresp_1stnew[i_ct].index_3D_frame2 ].position );
            imagePoints.push_back( vfeaturepoints_l[ vcorresp_1stnew[i_ct].match.queryIdx ].pt );
        }

        /* since we do not plan to optimize locations of camera and point cloud temporarily,
         * we shall not use pnp_mask. Nevertheless, we output it for any possible further use.
         */
        pose = usr_operator.standard_PnP(Pointcloud, imagePoints, cameraMatrix, pnp_mask);
    }

    /* Backward PnP 2D-3D
     * Not stable enough when there are a few points, mismatch is the main reason. */
    else if(algo == BPNP)
    {
        std::vector< cv::Point3f > Pointcloud;
        std::vector< cv::Point2f > imagePoints;
        cv::Mat pnp_mask, cameraMatrix = CamProjMat_l(Range(0,3), Range(0,3));

        for (int i_ct = 0; i_ct < vcorresp_2ndnew.size(); i_ct++)
        {
            Pointcloud.push_back( pframeTomap->cloudMap[ vcorresp_2ndnew[i_ct].index_3D_frame1 ].position );
            imagePoints.push_back( targetframe.vfeaturepoints_l[ vcorresp_2ndnew[i_ct].match.trainIdx ].pt );
        }

        /* since we do not plan to optimize locations of camera and point cloud temporarily,
         * we shall not use pnp_mask. Nevertheless, we output it for any possible further use.
         */
        pose = usr_operator.standard_PnP(Pointcloud, imagePoints, cameraMatrix, pnp_mask);

        pose = pose.inv();
    }

    /* Decompose essential matrix 2D-2D */
    else if(algo == DEE)
    {

    }


/// CHECK ESTIMATION OF POSE
    if( pose(0,0) == 0 )
    {
        cout << "Error:" << "\n"
             << "Fail to estimate pose. \n"
             << "System is LOST in Frame:"  << targetframe.name << endl;
        exit(EXIT_FAILURE);
    }

    /* There must be some mistakes when translation is larger than 2, since the system is
     * installed in a car or it is held by human, with a refreashing frequency of 10Hz */
    else if( pose(0, 3)*pose(0, 3) + pose(1, 3)*pose(1, 3) + pose(2, 3)*pose(2, 3) > REJECT_DISTANCE*REJECT_DISTANCE )
    {
        cout << "Error:" << "\n"
             << "Estimation of pose is unreliable. \n"
             << "System is Down in Frame:"  << targetframe.name << endl;
        cout << "ill pose is:" << pose << endl;
        cv::waitKey(0);
        pframeTomap->draw_Map();
        exit(EXIT_FAILURE);
    }
    else
    {
        targetframe.T_w2c = pose * T_w2c;
    }
}

void Frame::match_frames( Frame &targetframe )
{
    std::vector< cv::DMatch > vinterframe_matches, vinterframe_matchesinlier;
    SURFMatcher<cv::BFMatcher> matcher;

    matcher.match(descriptors_l, targetframe.descriptors_l, vinterframe_matches);

    find_inliers(vfeaturepoints_l, targetframe.vfeaturepoints_l, vinterframe_matches, vinterframe_matchesinlier);

    estimate_pose( vinterframe_matchesinlier, targetframe, PNP );
}

void Frame::reprojectInterFrameTo3D( Frame targetframe , std::vector<Mappoint> &v_mappoints )
{
}
