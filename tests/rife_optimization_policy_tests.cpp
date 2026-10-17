#include "nvof/rife_optimization_policy.hpp"
#include <stdexcept>
#include <iostream>
int main(){try{
    using nvof::rife_packed_motion_resolution;
    for(auto size:{std::pair{1920,1080},std::pair{2560,1440},std::pair{3440,1440},std::pair{3838,2160},std::pair{1440,2560}})
        if(rife_packed_motion_resolution(size.first,size.second))throw std::runtime_error("sub-UHD enabled E04");
    for(auto size:{std::pair{3840,1604},std::pair{3840,2160},std::pair{4096,2160},std::pair{7680,4320},std::pair{1604,3840}})
        if(!rife_packed_motion_resolution(size.first,size.second))throw std::runtime_error("UHD/cropped/portrait disabled E04");
    if(rife_packed_motion_resolution(0,4320)||rife_packed_motion_resolution(3840,-1))throw std::runtime_error("invalid source enabled E04");
    std::cout<<"PASS E04 source long-edge policy: FHD/QHD/off, cropped-UHD/portrait/8K/on\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
