#ifndef MAP_H
#define MAP_H

#include <vector>

#include "mappoint.h"

class Map
{
public:
    Map();

    Map( std::vector< Mappoint > pointcloud );

public:

    std::vector< Mappoint > cloudMap;

};




#endif
