#include <cstring>
#include "scene_fixture.hpp"
#include <iostream>
using namespace nvof;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(int argc,char**argv){try{
 require(argc==2,"Expected scene fixture binary");auto fixtures=read_scene_fixtures(argv[1]);unsigned labeled=0,cuts=0,wrong=0;SceneStats realCut{};
 for(auto& f:fixtures){SceneDetector detector;for(size_t i=1;i<f.frames.size();++i){auto s=scene_stats_grid(f.frames[i-1],f.frames[i]);auto d=detector.evaluate(s);
   if(f.labels[i]>=0){++labeled;cuts+=f.labels[i];if(d.cut!=bool(f.labels[i])){++wrong;std::cerr<<"Mismatch "<<f.name<<" frame "<<i<<" expected="<<f.labels[i]<<" actual="<<d.cut<<" norm="<<d.normalized<<" patch="<<d.patch_mean<<" baseline="<<d.baseline<<'\n';}if(f.labels[i]&&d.cut)realCut=s;}
  }}
 require(labeled>=389,"Missing labeled motion/cut regression cases");require(wrong==0,"Real source scene classification failed");
 SceneGrid base{},flash{},gain{},pan{},fade{};
 for(unsigned y=0;y<36;++y)for(unsigned x=0;x<64;++x){unsigned i=y*64+x;unsigned a=40+((x/4+y/3)%2)*90;base[i]=a;flash[i]=a+50;gain[i]=uint8_t(a*.6+80);pan[i]=40+(((x+4)/4+y/3)%2)*90;fade[i]=(a*3+120)/4;}
 for(auto other:{flash,gain,pan,fade}){SceneDetector d;require(!d.evaluate(scene_stats_grid(base,other)).cut,"Exposure/motion/fade must retain interpolation");}
 SceneDetector reset;auto expected=reset.evaluate(realCut);for(int i=0;i<20;++i)reset.evaluate(scene_stats_grid(base,pan));reset.reset();require(reset.evaluate(realCut).cut==expected.cut,"Seek history leaked into new pair");
 require(!classify_scene({}).cut,"Empty stats cannot be a scene cut");
 std::cout<<"PASS labeled="<<labeled<<" actual_cuts="<<cuts<<" wrong=0 exposure/gain/pan/fade/seek-reset\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
