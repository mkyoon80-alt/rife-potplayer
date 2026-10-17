// Exercise the same-module file-switch path: a fresh D3D device/buffer set
// reuses the cached TensorRT context, including cancellation during warmup.
#define wmain unused_rife_probe_main
#include "rife_probe.cpp"
#undef wmain

int wmain(int argc,wchar_t** argv) {
    timeBeginPeriod(1);
    try {
        require(argc>=2,"runtime required");
        const int cycles=argc>2?std::stoi(argv[2]):12;
        auto a=pattern(1920,1080,0,true),b=pattern(1920,1080,8,true);
        std::vector<std::vector<uint8_t>> reference;
        for(int file=0;file<cycles;++file) {
            Device dev;
            auto fa=dev.upload(a,1920,1080,0,true),fb=dev.upload(b,1920,1080,420000,true);
            auto e=engine(dev,argv[1]);
            auto first=e->copy(fa,true);
            require(e->startup_warmup()==2,"external startup warmup required");
            if(file%4!=3) {
                auto second=e->copy(fb,true);
                auto batch=e->interpolate_pair(first,second,{84000,168000,252000,336000});
                require(batch.frames.size()==4 && batch.quality.native_synthesized_mask==15,"startup inference absent");
                for(size_t i=0;i<batch.frames.size();++i) {
                    auto bytes=dev.read(batch.frames[i]);
                    if(reference.size()<4)reference.push_back(bytes);
                    else require(bytes==reference[i],"file switch changed pixels");
                }
            }
            e.reset();
            check(dev.d->GetDeviceRemovedReason(),"device after transition");
            std::cout<<"PASS file="<<file<<" cancel="<<(file%4==3)
                     <<" contexts="<<MotionSynthesizer::execution_create_count()<<std::endl;
        }
        timeEndPeriod(1);return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FAIL "<<e.what()<<std::endl;
        timeEndPeriod(1);return 1;
    }
}
