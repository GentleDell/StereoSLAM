#include "usrmath.h"

cv::Mat Usrmath::matched_ICP()
{

}

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

cv::Matx44d Usrmath::standard_PnP(std::vector< cv::Point3f > Pointcloud,
                                  std::vector< cv::Point2f > imagePoints,
                                  cv::Mat cameraMatrix, cv::Mat &pnpmask)
{
    bool pnpflag;
    cv::Mat rotation, translation, rotMat;

    pnpflag = cv::solvePnPRansac( Pointcloud, imagePoints, cameraMatrix, cv::Mat(),
                                  rotation, translation, false, 100, 8.0, 0.99, pnpmask, cv::SOLVEPNP_ITERATIVE);

    cv::Rodrigues(rotation, rotMat);

    cv::Mat temp_h, temp_v;
    cv::Mat lastline = (cv::Mat_<double>(1, 4) << 0, 0, 0, 1);
    cv::hconcat(rotMat, translation, temp_h);
    cv::vconcat(temp_h, lastline, temp_v);

    return cv::Matx44d(temp_v);

}

cv::Mat Usrmath::decompose_essenMat()
{

}
