#ifndef MAP_H
#define MAP_H

#include <vector>

#include "frame.h"
#include "mappoint.h"

class Map
{
public:
    Map();

public:

    std::vector< Mappoint > cloudMap;

    std::vector< Frame > frameMap;

};




#endif
