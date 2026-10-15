#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain
#include <map>
#include <set>
void diagnostics_test(Device& dev,const std::filesystem::path& runtime){
    auto e=engine(dev,runtime);uint64_t lastPair=0;unsigned gpu=0,cuda=0;
    std::set<std::pair<uint64_t,float>> seen;
    auto consume=[&](const RifeTiming& t){
        require(!t.cuda_invalid&&!t.cuda_skipped&&!t.d3d_invalid&&!t.d3d_skipped,"Diagnostic samples invalid/skipped");
        auto add=[&](const RifePhaseTiming& q){if(!q.cuda_valid)return;require(!q.cuda_pending&&q.inference_gpu_ms>0&&q.pts==400000,"Invalid CUDA result");require(seen.emplace(q.pair,q.phase).second,"CUDA sample reported twice");++cuda;};
        for(unsigned i=0;i<t.phase_count;++i)add(t.phases[i]);
        for(unsigned i=0;i<t.completed_phase_count;++i){require(t.completed_phases[i].pair<t.pair,"Delayed CUDA attribution failed");add(t.completed_phases[i]);}
        for(unsigned i=0;i<t.gpu_count;++i){auto& q=t.gpu[i];require(q.pair<t.pair&&q.gpu_ms>=0,"Delayed D3D attribution failed");++gpu;}
    };
    for(auto spec:std::vector<std::array<int,4>>{{1920,1080,0,0},{3840,2160,1,15},{1080,1920,0,0}}){
        int w=spec[0],h=spec[1];auto a=pattern(w,h,0,spec[2]),b=pattern(w,h,8,spec[2]);auto fa=dev.upload(a,w,h,0,spec[2],spec[3]),fb=dev.upload(b,w,h,400000,spec[2],spec[3]);
        e->enable_timing(false);auto original=e->interpolate_pair(fa,fb,{100000,200000,300000});std::vector<std::vector<uint8_t>> refs;for(auto& f:original.frames)refs.push_back(dev.read(f));
        require(!e->pair_timing().rife,"Disabled diagnostics produced data");e->reset();e->enable_timing(true,true);
        for(int k=0;k<4;++k){auto batch=e->interpolate_pair(fa,fb,{100000,200000,300000});dev.wait();auto t=e->pair_timing().rife?*e->pair_timing().rife:RifeTiming{};
            require(t.enabled&&t.phase_count==3&&t.pair>lastPair,"Per-pair profiling identity failed");lastPair=t.pair;consume(t);
            for(unsigned i=0;i<t.phase_count;++i){auto& q=t.phases[i];require((q.cuda_valid||q.cuda_pending)&&q.pair==t.pair,"Current CUDA sample missing");require(dev.read(batch.frames[i])==refs[i],"Profiling changed output pixels");}
            e->reset();
        }
        auto same=fa;same.pts=400000;auto calls=MotionSynthesizer::inference_count();e->interpolate_pair(fa,same,{200000});dev.wait();
        require(MotionSynthesizer::inference_count()==calls,"Still frame inferred");if(e->pair_timing().rife){require(e->pair_timing().rife->phase_count==0,"Still frame timing invented");consume(*e->pair_timing().rife);}
        e->enable_timing(false);e->interpolate_pair(fa,fb,{200000});require(!e->pair_timing().rife,"Diagnostic disable failed");
    }
    require(gpu>=21&&cuda==36,"Missing delayed CUDA/D3D measurements");std::cout<<"PASS profiling pixels unchanged, CUDA="<<cuda<<" delayed_D3D="<<gpu<<" seek/toggle/still/HDR\n";
}
void graph_counts(){std::cout<<"GRAPH launches="<<MotionSynthesizer::graph_launch_count()<<" builds="<<MotionSynthesizer::graph_build_count()<<" fallbacks="<<MotionSynthesizer::graph_fallback_count()<<std::endl;}
int wmain(int argc,wchar_t**argv){timeBeginPeriod(1);try{
    require(argc>=3,"rife_profile runtime --self-test OR fixture-folder output-prefix round");Device dev;
    if(std::wstring(argv[2])==L"--self-test"){diagnostics_test(dev,argv[1]);graph_counts();timeEndPeriod(1);return 0;}
    require(argc>=5,"fixture-folder and output-prefix required");const std::filesystem::path folder=argv[2],prefix=argv[3];int round=std::stoi(argv[4]);auto e=engine(dev,argv[1]);
    std::ifstream manifest(folder/L"manifest.txt");require(bool(manifest),"Manifest missing");std::ofstream cpu(prefix.wstring()+L"-cpu.csv"),gpu(prefix.wstring()+L"-gpu.csv"),pairs(prefix.wstring()+L"-pairs.csv");
    cpu<<"case,phases,mode,pair,pts,phase,width,height,cuda_valid,trt_gpu,map_stream,unmap_stream,time_cpu,device_cpu,map_cpu,bind_cpu,enqueue_cpu,unmap_cpu,sync_cpu,composite_cpu,event_read_cpu,phase_cpu,status_query_cpu,graph_replay,graph_created\n";
    gpu<<"case,phases,mode,pair,pts,phase,width,height,kind,gpu_ms\n";
    pairs<<"case,phases,mode,iteration,pair,wall_ms,scene_cut,repeated,phases_measured,initialize_cpu,guide_cpu,collect_cpu,cuda_invalid,d3d_invalid,d3d_skipped,cuda_skipped\n";
    std::string id;int w,h,ten,transfer,expected;
    while(manifest>>id>>w>>h>>ten>>transfer>>expected){
        size_t bytes=size_t(w)*h*3/2*(ten?2:1);auto raw=load(folder/id/L"pair.raw",bytes*2);std::vector<uint8_t>a(raw.begin(),raw.begin()+bytes),b(raw.begin()+bytes,raw.end());
        auto fa=dev.upload(a,w,h,0,ten,transfer),fb=dev.upload(b,w,h,400000,ten,transfer);
        // Retain attribution across mode boundaries: a completed CUDA/D3D
        // sample can arrive in the following block, including after a seek.
        std::map<uint64_t,std::pair<int,int>> kept;
        auto emit=[&](const RifePhaseTiming& q){auto found=kept.find(q.pair);if(!q.cuda_valid||found==kept.end())return;auto [ph,mode]=found->second;
            cpu<<id<<','<<ph<<','<<mode<<','<<q.pair<<','<<q.pts<<','<<q.phase<<','<<q.width<<','<<q.height<<','<<q.cuda_valid<<','<<q.inference_gpu_ms<<','<<q.map_stream_ms<<','<<q.unmap_stream_ms<<','<<q.time_submit_cpu_ms<<','<<q.device_cpu_ms<<','<<q.map_cpu_ms<<','<<q.bind_cpu_ms<<','<<q.enqueue_cpu_ms<<','<<q.unmap_cpu_ms<<','<<q.sync_cpu_ms<<','<<q.composite_submit_cpu_ms<<','<<q.event_read_cpu_ms<<','<<q.total_cpu_ms<<','<<q.status_query_cpu_ms<<','<<q.graph_replayed<<','<<q.graph_created<<'\n';};
        for(int phases:{1,3}){auto pts=phases==1?std::vector<int64_t>{200000}:std::vector<int64_t>{100000,200000,300000};
            for(int mode:(round%2?std::vector<int>{0,2,3,3,2,0}:std::vector<int>{3,2,0,0,2,3})){
                e->reset();e->enable_timing(mode!=0,mode>=2,mode==2?32:1);
                for(int i=0;i<52;++i){auto start=Clock::now();auto batch=e->interpolate_pair(fa,fb,pts);dev.wait();double wall=elapsed(start);auto t=e->pair_timing().rife?*e->pair_timing().rife:RifeTiming{};
                    if(expected>=0)require(batch.quality.scene_cut==bool(expected),"Scene decision changed");
                    if(i>=12){if(t.enabled)kept[t.pair]={phases,mode};pairs<<id<<','<<phases<<','<<mode<<','<<i<<','<<t.pair<<','<<wall<<','<<batch.quality.scene_cut<<','<<batch.quality.identical_warp_skipped<<','<<t.phase_count<<','<<t.initialize_cpu_ms<<','<<t.guide_submit_cpu_ms<<','<<t.query_collect_cpu_ms<<','<<t.cuda_invalid<<','<<t.d3d_invalid<<','<<t.d3d_skipped<<','<<t.cuda_skipped<<'\n';}
                    for(unsigned j=0;j<t.phase_count;++j)emit(t.phases[j]);for(unsigned j=0;j<t.completed_phase_count;++j)emit(t.completed_phases[j]);
                    for(unsigned j=0;j<t.gpu_count;++j){auto& q=t.gpu[j];auto found=kept.find(q.pair);if(found==kept.end())continue;auto [ph,m]=found->second;gpu<<id<<','<<ph<<','<<m<<','<<q.pair<<','<<q.pts<<','<<q.phase<<','<<q.width<<','<<q.height<<','<<unsigned(q.kind)<<','<<q.gpu_ms<<'\n';}
                }
            }
        }
        cpu.flush();gpu.flush();pairs.flush();std::cout<<id<<" completed\n"<<std::flush;
    }
    require(MotionSynthesizer::engine_load_count()==1,"Engine reloaded");std::cout<<"PASS engine_loads=1\n";graph_counts();timeEndPeriod(1);return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';timeEndPeriod(1);return 1;}}
