#pragma once
#include "nvof/scene_cut.hpp"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
struct SceneFixture{std::string name;std::vector<nvof::SceneGrid> frames;std::vector<int32_t> labels;};
inline std::vector<SceneFixture> read_scene_fixtures(const char* path){
 std::ifstream in(path,std::ios::binary);char magic[8]{};in.read(magic,8);if(std::string(magic,6)!="SCENE2")throw std::runtime_error("Invalid scene fixtures");
 uint32_t count=0;in.read(reinterpret_cast<char*>(&count),4);if(count>1000)throw std::runtime_error("Invalid window count");std::vector<SceneFixture> fixtures;
 for(unsigned w=0;w<count;++w){char name[40]{};uint32_t n=0;in.read(name,40);in.read(reinterpret_cast<char*>(&n),4);if(n>100000)throw std::runtime_error("Invalid frame count");SceneFixture f;f.name=std::string(name,strnlen(name,40));f.frames.resize(n);f.labels.resize(n);
  for(unsigned i=0;i<n;++i){in.read(reinterpret_cast<char*>(&f.labels[i]),4);in.read(reinterpret_cast<char*>(f.frames[i].data()),f.frames[i].size());}
  if(!in)throw std::runtime_error("Truncated scene fixtures");fixtures.push_back(std::move(f));
 }return fixtures;
}
