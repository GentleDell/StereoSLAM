#include "map.h"

Map::Map(){

}

void Map::draw_Map()
{
    stringstream ss;

    viz::Viz3d myWindow("Global Map");

    // Add coordinate axes
    myWindow.showWidget("Coordinate", viz::WCoordinateSystem());

    // Construct a cube widget
    viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);

    // Display widget (update if already displayed)
    myWindow.showWidget("Cube Widget", cube_widget);

    for(int i_ct = 0; i_ct < frameMap.size(); i_ct++)
    {
        // Construct pose
        Affine3f pose(frameMap[i_ct].T_w2c);

        myWindow.setWidgetPose("Cube Widget", pose);

        std::vector< cv::Point3f > pointcloud;
        for (int j_ct = 0; j_ct < frameMap[i_ct].vMappoints_indexnum.size(); j_ct++){
              pointcloud.push_back( cloudMap[ frameMap[i_ct].vMappoints_indexnum[j_ct] ].position );
        }
        cv::viz::WCloud cloud_widget(pointcloud, cv::viz::Color::green());

        // Visualize widget
        ss.clear();
        ss.str("");
        ss << "features " << i_ct;

        myWindow.showWidget(ss.str(), cloud_widget, frameMap[i_ct].T_w2c);

        myWindow.spinOnce(500, true);
    }
}
