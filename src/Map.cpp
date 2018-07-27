#include "map.h"

Map::Map(){

}

void Map::draw_Map()
{
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

        myWindow.spinOnce(500, true);
    }
}
