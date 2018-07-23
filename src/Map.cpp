#include "map.h"

Map::Map(){

}

Map::Map( std::vector< Mappoint > pointcloud )
{
    cloudMap = pointcloud;
}
