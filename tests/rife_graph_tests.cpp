#include <array>
#include <vector>
#include <string>
#include <atomic>
#include <stdexcept>
#include <iostream>
using cudaGraphExec_t=void*;using cudaGraph_t=void*;using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaStreamCaptureModeThreadLocal=1;
std::atomic<uint64_t> graph_launches{0},graph_builds{0},graph_fallbacks{0};
struct Cuda {
    int begin_error=0,end_error=0,instantiate_error=0,launch_error=0;
    unsigned begin=0,end=0,destroy=0,exec_destroy=0,sync=0,launch=0;
    int cudaStreamSynchronize(void*){++sync;return 0;}
    int cudaGraphExecDestroy(void*){++exec_destroy;return 0;}
    int cudaGraphLaunch(void*,void*){++launch;return launch_error;}
    int cudaStreamBeginCapture(void*,int){++begin;return begin_error;}
    int cudaStreamEndCapture(void*,void** graph){++end;*graph=reinterpret_cast<void*>(1);return end_error;}
    int cudaGraphInstantiateWithFlags(void** graph,void*,unsigned long long){*graph=reinterpret_cast<void*>(2);return instantiate_error;}
    int cudaGraphDestroy(void*){++destroy;return 0;}
    void check(int error,const char* text){if(error)throw std::runtime_error(text);}
};
struct Context {unsigned enqueue=0,bind=0;bool accepted=true;bool enqueueV3(void*){++enqueue;return accepted;}bool setTensorAddress(const char*,void*){++bind;return true;}};
#include "../src/rife_graph.inl"
void expect(bool x){if(!x)throw std::runtime_error("Graph fallback/lifetime invariant failed");}
int main(){try{
    const std::array<void*,2> a{reinterpret_cast<void*>(10),reinterpret_cast<void*>(20)},b{reinterpret_cast<void*>(30),reinterpret_cast<void*>(40)};
    {Cuda cu;Context ctx;RifeGraph g;g.run(cu,ctx,nullptr,a,true);expect(ctx.enqueue==1&&!g.executable);g.run(cu,ctx,nullptr,a,true);g.run(cu,ctx,nullptr,a,true);
        expect(ctx.enqueue==2&&ctx.bind==2&&cu.launch==2&&cu.begin==1);g.run(cu,ctx,nullptr,b,true);
        expect(cu.sync==1&&cu.exec_destroy==1&&!g.executable&&ctx.bind==4&&ctx.enqueue==3);g.run(cu,ctx,nullptr,b,true);expect(cu.begin==2&&g.executable);g.clear(cu);expect(!g.executable&&cu.exec_destroy==2);}
    for(int failure=0;failure<3;++failure){Cuda cu;Context ctx;RifeGraph g;if(failure==0)cu.begin_error=1;if(failure==1)cu.end_error=1;if(failure==2)cu.instantiate_error=1;
        g.run(cu,ctx,nullptr,a,true);g.run(cu,ctx,nullptr,a,true);expect(g.disabled&&!g.executable);g.run(cu,ctx,nullptr,a,true);
        expect(cu.begin==1&&cu.end==(failure?1u:0u)&&cu.launch==0&&ctx.enqueue==(failure?4u:3u));if(failure==2)expect(cu.exec_destroy==1);}
    {Cuda cu;Context ctx;RifeGraph g;for(int i=0;i<4;++i)g.run(cu,ctx,nullptr,a,false);expect(cu.begin==0&&ctx.enqueue==4);}
    {Cuda cu;Context ctx;RifeGraph g;g.run(cu,ctx,nullptr,a,true);g.run(cu,ctx,nullptr,a,true);cu.launch_error=1;bool threw=false;try{g.run(cu,ctx,nullptr,a,true);}catch(...){threw=true;}
        expect(threw&&ctx.enqueue==2);g.clear(cu);}
    std::cout<<"PASS graph warmup/reuse, changed-address fence, unsupported capture fallback, partial instantiate cleanup, no duplicate inference after launch error\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
