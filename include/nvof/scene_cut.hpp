#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <initializer_list>
namespace nvof {
inline constexpr unsigned scene_grid_width=64,scene_grid_height=36,scene_grid_count=2304;
// Layout shared with scene.hlsl. Only these aggregates, not the temporary
// sampled image grid, are copied to the CPU during playback.
struct SceneStats {
 uint32_t samples=0,absolute_sum=0;int32_t signed_sum=0;
 uint32_t squared_sum=0,changed=0,histogram_a[32]{},histogram_b[32]{};
 uint32_t sum_a=0,sum_b=0,square_a=0,square_b=0,product=0;
 uint32_t edge_a=0,edge_b=0;int32_t edge_product=0;
 uint32_t patch_weight=0;int32_t patch_sum=0;uint32_t patch_matched=0;
};
static_assert(sizeof(SceneStats)==80*sizeof(uint32_t));
inline constexpr unsigned scene_identity_index=80,scene_grid_offset=81;
struct SceneDecision {
 bool cut=false;double difference=0,coverage=0,histogram_distance=0,variation=0;
 double contrast=0,normalized=0,edge_correlation=0,correlation=0,palette=0,edge_balance=0;
 double patch_mean=1,patch_fraction=1,baseline=0;
};
inline SceneDecision scene_features(const SceneStats& s){
 SceneDecision d{};if(!s.samples)return d;const double n=s.samples;
 d.difference=s.absolute_sum/n;d.coverage=s.changed/n;double mean=s.signed_sum/n;
 d.variation=std::sqrt((std::max)(0.,s.squared_sum/n-mean*mean));
 double va=(std::max)(0.,s.square_a/n-std::pow(s.sum_a/n,2)),vb=(std::max)(0.,s.square_b/n-std::pow(s.sum_b/n,2));
 d.contrast=std::sqrt((va+vb)*.5);
 d.normalized=d.difference/(std::max)(8.,d.contrast);
 d.correlation=(s.product/n-(s.sum_a/n)*(s.sum_b/n))/(std::max)(1.,std::sqrt(va*vb));
 d.edge_correlation=double(s.edge_product)/(std::max)(1.,std::sqrt(double(s.edge_a)*s.edge_b));
 d.edge_balance=double((std::min)(s.edge_a,s.edge_b))/(std::max)(1u,(std::max)(s.edge_a,s.edge_b));
 unsigned first=0,second=0;
 for(unsigned i=0;i<32;++i){
  d.histogram_distance+=std::abs(double(s.histogram_a[i])-s.histogram_b[i]);
  if(s.histogram_a[i]>s.histogram_a[first])first=i;
 }
 d.histogram_distance/=2*n;
 // Preserve animated two-tone shapes, but do not mistake a dark scene's
 // adjacent luminance bins for a black/white graphic palette.
 unsigned secondCount=0;for(unsigned i=0;i<32;++i)if(std::abs(int(i)-int(first))>=12&&s.histogram_a[i]>=secondCount){second=i;secondCount=s.histogram_a[i];}
 unsigned a0=0,a1=0,b0=0,b1=0;
 for(unsigned i=0;i<32;++i){if(std::abs(int(i)-int(first))<=2){a0+=s.histogram_a[i];b0+=s.histogram_b[i];}if(std::abs(int(i)-int(second))<=2){a1+=s.histogram_a[i];b1+=s.histogram_b[i];}}
 if((std::min)({a0,a1,b0,b1})>=.05*n)d.palette=(std::min)(a0+a1,b0+b1)/n;
 if(s.patch_weight){d.patch_mean=double(s.patch_sum)/(1024.*s.patch_weight);d.patch_fraction=double(s.patch_matched)/s.patch_weight;}
 return d;
}
inline bool scene_candidate(const SceneDecision& d){
 return d.difference>=6.&&d.normalized>=.65&&d.histogram_distance>=.10&&d.edge_correlation<.35&&d.correlation<.90&&!(d.palette>=.80&&d.normalized<2.)&&!(d.edge_correlation>=.25&&d.correlation>=.60);
}
inline SceneDecision classify_scene(const SceneStats& s){
 auto d=scene_features(s);
 d.cut=scene_candidate(d)&&(d.patch_mean<.74||(d.histogram_distance>=.20&&d.patch_mean<.80&&d.patch_fraction<.60));
 // Defocus/foreground occlusion destroys fine edges but retains broad local
 // structure. Do not insert a cut solely because edge energy collapsed.
 if(d.edge_balance<.12&&d.patch_mean>=.70&&d.histogram_distance<.60)d.cut=false;
 return d;
}
// Causal history: no future frame, sleep, inference or texture retention.
// Repeated animation cells do not dilute the recent motion baseline to zero.
class SceneDetector {
 std::array<double,5> motion_{};unsigned count_=0,idle_=0;
public:
 void reset()noexcept{motion_={};count_=idle_=0;}
 SceneDecision evaluate(const SceneStats& s){
  auto d=classify_scene(s);auto sorted=motion_;std::sort(sorted.begin(),sorted.begin()+count_);
  if(count_)d.baseline=count_%2?sorted[count_/2]:(sorted[count_/2-1]+sorted[count_/2])*.5;
  const double ratio=d.patch_mean>=.74?6.:d.patch_mean>=.65?3.:1.5;
  d.cut=d.cut&&d.normalized>=(std::max)(.10,d.baseline)*ratio;
  // A cold/seek pair has no preceding motion evidence. Preserve the existing
  // weak-histogram veto for high-contrast graphics/fast moves, or when most
  // local patches still match strongly. Low-contrast similar-palette cuts
  // must not be vetoed just because their average patch match remains high.
  // Contrast is luma standard deviation in the existing 8-bit scene grid;
  // this uses already-collected statistics, with no extra GPU work or wait.
  if(!count_&&d.patch_mean>=.70&&d.histogram_distance<.15&&
     (d.contrast>=32.||d.patch_fraction>=.60))d.cut=false;
  if(d.normalized<=.08){if(++idle_>=12)reset();}
  else{idle_=0;if(!d.cut&&(d.normalized<.4||(d.edge_correlation<.4&&d.correlation<.85))){
   if(count_==motion_.size()){std::move(motion_.begin()+1,motion_.end(),motion_.begin());--count_;}
   motion_[count_++]=d.normalized;
  }}
  return d;
 }
};
inline void add_scene_sample(SceneStats& s,unsigned a,unsigned b){
 int delta=int(b)-int(a);unsigned absolute=unsigned(std::abs(delta));
 ++s.samples;s.absolute_sum+=absolute;s.signed_sum+=delta;s.squared_sum+=unsigned(delta*delta);
 s.changed+=absolute>=32;++s.histogram_a[a>>3];++s.histogram_b[b>>3];
 s.sum_a+=a;s.sum_b+=b;s.square_a+=a*a;s.square_b+=b*b;s.product+=a*b;
}
using SceneGrid=std::array<uint8_t,scene_grid_count>;
inline void scene_patch_stats(SceneStats& s,const SceneGrid& a,const SceneGrid& b){
 if(!scene_candidate(scene_features(s)))return;
 for(unsigned direction=0;direction<2;++direction){
  const auto& aa=direction?b:a;const auto& bb=direction?a:b;
  const double sum=direction?s.sum_b:s.sum_a,sq=direction?s.square_b:s.square_a;
  const double minimum=(std::max)(3.,.12*std::sqrt((std::max)(0.,sq/s.samples-std::pow(sum/s.samples,2))));
  for(int y=0;y<=24;y+=6)for(int x=0;x<=56;x+=8){
   double sa=0,saa=0;for(int j=0;j<7;++j)for(int i=0;i<7;++i){double v=aa[(y+j)*64+x+i];sa+=v;saa+=v*v;}
   double energy=saa-sa*sa/49.,stddev=std::sqrt((std::max)(0.,energy/49.));if(stddev<minimum)continue;
   double best=-1.;
   for(int yy=(std::max)(0,y-6);yy<=(std::min)(29,y+6);++yy)for(int xx=(std::max)(0,x-10);xx<=(std::min)(57,x+10);++xx){
    double sb=0,sbb=0,sab=0;
    for(int j=0;j<7;++j)for(int i=0;i<7;++i){double av=aa[(y+j)*64+x+i],bv=bb[(yy+j)*64+xx+i];sb+=bv;sbb+=bv*bv;sab+=av*bv;}
    double e=sbb-sb*sb/49.;double correlation=e>1e-5?(sab-sa*sb/49.)/std::sqrt(energy*e):0.;best=(std::max)(best,(std::min)(1.,correlation));
   }
   unsigned weight=unsigned(std::lround((std::min)(stddev,30.)*256.));s.patch_weight+=weight;
   s.patch_sum+=int32_t(std::lround(best*weight*1024.));if(best>.80)s.patch_matched+=weight;
  }
 }
}
inline SceneStats scene_stats_grid(const SceneGrid& a,const SceneGrid& b){
 SceneStats s{};for(unsigned y=0;y<36;++y)for(unsigned x=0;x<64;++x){unsigned k=y*64+x;add_scene_sample(s,a[k],b[k]);
  for(unsigned step:{1u,64u}){if((step==1&&x==63)||(step==64&&y==35))continue;int da=int(a[k+step])-a[k],db=int(b[k+step])-b[k];s.edge_a+=da*da;s.edge_b+=db*db;s.edge_product+=da*db;}
 }
 scene_patch_stats(s,a,b);return s;
}
inline SceneStats scene_stats_cpu(const uint8_t* a,const uint8_t* b,int width,int height){
 SceneGrid ga{},gb{};for(unsigned y=0;y<36;++y)for(unsigned x=0;x<64;++x){unsigned va=0,vb=0;
  for(unsigned oy:{1u,3u})for(unsigned ox:{1u,3u}){unsigned px=(x*4+ox)*unsigned(width)/256,py=(y*4+oy)*unsigned(height)/144;va+=a[size_t(py)*width+px];vb+=b[size_t(py)*width+px];}
  ga[y*64+x]=(va+2)/4;gb[y*64+x]=(vb+2)/4;
 }return scene_stats_grid(ga,gb);
}
}
