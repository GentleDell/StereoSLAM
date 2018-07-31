#include "usrmath.h"

cv::Matx44d Usrmath::standard_ICP(cv::Mat srcPC, cv::Mat dstPC, cv::Matx44d pose)
{
    int flag;
    cv::ppf_match_3d::ICP ICP_operator;
    cv::ppf_match_3d::Pose3D pose3D;
    cv::ppf_match_3d::Pose3DPtr p_pose3D;
    std::vector<cv::ppf_match_3d::Pose3DPtr> v_p_pose3D;

    pose3D.updatePose(pose);
    p_pose3D = pose3D.clone();
    v_p_pose3D.push_back(p_pose3D);

    flag = ICP_operator.registerModelToScene( srcPC, dstPC, v_p_pose3D );

    if(flag)
    {
        cout << "ERROR:\n"
             << "ICP failed!  file: usrmath.cpp" << endl;

        v_p_pose3D[0]->pose = cv::Matx44d::zeros();
    }   

    return v_p_pose3D[0]->pose;
}

cv::Matx44d Usrmath::matched_ICP(std::vector< cv::Point3f > vsrcPointCloud, std::vector< cv::Point3f > vdstPointCloud, int flag)
{
    int num_matches = vsrcPointCloud.size();
    cv::Mat_<double> srccentroid(3,1,0.0), dstcentroid(3,1,0.0);
    cv::Point3f decentr_src(0,0,0), decentr_dst(0,0,0);

    for(int i_ct = 0; i_ct < num_matches; i_ct++)
    {
        srccentroid.operator ()(0,0) += vsrcPointCloud[i_ct].x;
        srccentroid.operator ()(1,0) += vsrcPointCloud[i_ct].y;
        srccentroid.operator ()(2,0) += vsrcPointCloud[i_ct].z;

        dstcentroid.operator ()(0,0) += vdstPointCloud[i_ct].x;
        dstcentroid.operator ()(1,0) += vdstPointCloud[i_ct].y;
        dstcentroid.operator ()(2,0) += vdstPointCloud[i_ct].z;
    }
    srccentroid.operator ()(0,0) /= num_matches;
    srccentroid.operator ()(1,0) /= num_matches;
    srccentroid.operator ()(2,0) /= num_matches;

    dstcentroid.operator ()(0,0) /= num_matches;
    dstcentroid.operator ()(1,0) /= num_matches;
    dstcentroid.operator ()(2,0) /= num_matches;

    cv::Mat_<double> W(3,3,0.0);
    for(int i_ct = 0; i_ct < num_matches; i_ct++)
    {
        decentr_src.x = vsrcPointCloud[i_ct].x - srccentroid.operator ()(0,0);
        decentr_src.y = vsrcPointCloud[i_ct].y - srccentroid.operator ()(1,0);
        decentr_src.z = vsrcPointCloud[i_ct].z - srccentroid.operator ()(2,0);

        decentr_dst.x = vdstPointCloud[i_ct].x - dstcentroid.operator ()(0,0);
        decentr_dst.y = vdstPointCloud[i_ct].y - dstcentroid.operator ()(1,0);
        decentr_dst.z = vdstPointCloud[i_ct].z - dstcentroid.operator ()(2,0);

        W.operator ()(0,0) += decentr_src.x*decentr_src.x;
        W.operator ()(0,1) += decentr_src.x*decentr_src.y;
        W.operator ()(0,2) += decentr_src.x*decentr_src.z;
        W.operator ()(1,0) += decentr_src.y*decentr_src.x;
        W.operator ()(1,1) += decentr_src.y*decentr_src.y;
        W.operator ()(1,2) += decentr_src.y*decentr_src.z;
        W.operator ()(2,0) += decentr_src.z*decentr_src.x;
        W.operator ()(2,1) += decentr_src.z*decentr_src.y;
        W.operator ()(2,2) += decentr_src.z*decentr_src.z;
    }

    cv::Mat U,Sig,Vt;
    cv::Mat rotation, translation;
    cv::SVD::compute(W, Sig, U, Vt);
    rotation = U*Vt;
    translation = srccentroid - rotation*dstcentroid;

    cv::Mat temp_h, temp_v;
    cv::Mat lastline = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
    cv::hconcat(rotation, translation, temp_h);
    cv::vconcat(temp_h, lastline, temp_v);

    return cv::Matx44d(temp_v);
}

cv::Matx44d Usrmath::standard_PnP(std::vector< cv::Point3f > vPointcloud,
                                  std::vector< cv::Point2f > vimagePoints,
                                  cv::Mat cameraMatrix, cv::Mat &pnpmask)
{
    bool pnpflag;
    cv::Mat rotation, translation, rotMat;

    if(vPointcloud.size() < 4)
    {
        cout << "ERROR:\n"
             << "No enough 2D-3D matches for PnP solver!\n"
             << "Now please try BPnP." << endl;

        return cv::Matx44d::zeros();
    }
    else
    {
        pnpflag = cv::solvePnPRansac( vPointcloud, vimagePoints, cameraMatrix, cv::Mat(),
                                      rotation, translation, false, 100, 8.0, 0.99, pnpmask, cv::SOLVEPNP_ITERATIVE);

        cv::Rodrigues(rotation, rotMat);

        cv::Mat temp_h, temp_v;
        cv::Mat lastline = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
        cv::hconcat(rotMat, translation, temp_h);
        cv::vconcat(temp_h, lastline, temp_v);

        return cv::Matx44d(temp_v);
    }
}

cv::Mat Usrmath::decompose_essenMat()
{

}
