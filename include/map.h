#ifndef MAP_H
#define MAP_H

#include <vector>

#include <opencv2/viz/vizcore.hpp>

#include "frame.h"
#include "mappoint.h"

class Map
{
public:
    Map();

    void draw_Map();

public:

    std::vector< Mappoint > cloudMap;

    std::vector< Frame > frameMap;

};




#endif
