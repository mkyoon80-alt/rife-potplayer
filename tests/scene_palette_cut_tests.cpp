#include "scene_fixture.hpp"
#include <iostream>
void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**argv){try{
 require(argc==2,"Expected scene regression fixture");auto fixtures=read_scene_fixtures(argv[1]);unsigned labeled=0,cuts=0,fresh=0;
 for(auto&f:fixtures){nvof::SceneDetector detector;
  for(size_t i=1;i<f.frames.size();++i){auto stats=nvof::scene_stats_grid(f.frames[i-1],f.frames[i]);auto result=detector.evaluate(stats);
   if(f.labels[i]>=0){require(result.cut==bool(f.labels[i]),"Sequence scene classification changed");++labeled;cuts+=f.labels[i];}
   if(f.labels[i]==1){nvof::SceneDetector cold;require(cold.evaluate(stats).cut,"Similar-palette cut lost after opening/seek");++fresh;
    cold.reset();for(int k=0;k<20;++k)cold.evaluate(nvof::scene_stats_grid(f.frames[i-1],f.frames[i-1]));require(cold.evaluate(stats).cut,"Still history hid real cut");}
  }
 }
 require(labeled==194&&cuts==6&&fresh==6,"Incomplete real-source regression fixture");
 std::cout<<"PASS labeled="<<labeled<<" cuts="<<cuts<<" fresh="<<fresh<<" still/seek\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
