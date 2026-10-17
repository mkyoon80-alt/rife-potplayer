// No diagnostic synchronization. Both APIs use bounded pools and ready-only
// collection; delayed samples retain the original pair / PTS / phase.
struct RifeProfiler {
    using Clock=std::chrono::steady_clock;
    ID3D11Device* d;ID3D11DeviceContext* c;Cuda* cu=nullptr;cudaStream_t stream=nullptr;
    uint64_t serial=0;RifeTiming data{};bool event_failed=false;
    struct CudaSlot {std::array<cudaEvent_t,6> events{};bool pending=false,valid=true;RifePhaseTiming sample{};};
    std::array<CudaSlot,128> cuda_slots{};
    std::array<CudaSlot*,32> active{};
    struct Slot {ComPtr<ID3D11Query> disjoint,start,stop;bool pending=false;RifeGpuTiming sample{};};
    std::array<Slot,128> slots{};
    RifeProfiler(ID3D11Device* device,ID3D11DeviceContext* context):d(device),c(context){}
    static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
    // True means consumed (including an invalid sample); false means not ready.
    bool read_cuda(CudaSlot& slot)noexcept{
        auto started=Clock::now();auto& sample=slot.sample;
        if(slot.valid){
            auto ready=cu->cudaEventQuery(slot.events[5]);
            if(ready==cudaErrorNotReady)return false;
            if(ready!=cudaSuccess)slot.valid=false;
        }
        if(slot.valid){
            float values[3]{};
            for(unsigned i=0;i<3;++i)if(cu->cudaEventElapsedTime(&values[i],slot.events[i*2],slot.events[i*2+1])!=cudaSuccess)slot.valid=false;
            if(slot.valid){sample.map_stream_ms=values[0];sample.inference_gpu_ms=values[1];sample.unmap_stream_ms=values[2];}
            if(slot.valid){float copy=0;if(cu->cudaEventElapsedTime(&copy,slot.events[3],slot.events[4])!=cudaSuccess)slot.valid=false;else sample.copy_gpu_ms=sample.copy_count?copy:0;}
        }
        sample.cuda_valid=slot.valid;sample.cuda_pending=false;
        sample.event_read_cpu_ms=ms(started);slot.pending=false;
        if(!slot.valid)++data.cuda_invalid;return true;
    }
    void begin_pair(bool enabled,int64_t pts){
        ++serial;if(!enabled){data.enabled=false;return;}
        data={};data.enabled=true;data.pair=serial;data.pts=pts;active={};auto started=Clock::now();
        if(cu)for(auto& slot:cuda_slots)if(slot.pending&&read_cuda(slot)){
            if(data.completed_phase_count<data.completed_phases.size())data.completed_phases[data.completed_phase_count++]=slot.sample;
            else ++data.cuda_skipped;
        }
        for(auto& slot:slots){
            if(!slot.pending)continue;
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT frequency{};
            auto hr=c->GetData(slot.disjoint.Get(),&frequency,sizeof(frequency),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if(hr==S_FALSE)continue;
            if(FAILED(hr)||frequency.Disjoint||!frequency.Frequency){slot.pending=false;++data.d3d_invalid;continue;}
            UINT64 a=0,b=0;
            auto ha=c->GetData(slot.start.Get(),&a,sizeof(a),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            auto hb=c->GetData(slot.stop.Get(),&b,sizeof(b),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if(ha==S_FALSE||hb==S_FALSE)continue;
            slot.pending=false;if(FAILED(ha)||FAILED(hb)||b<a){++data.d3d_invalid;continue;}
            slot.sample.gpu_ms=double(b-a)*1000.0/double(frequency.Frequency);
            if(data.gpu_count<data.gpu.size())data.gpu[data.gpu_count++]=slot.sample;else ++data.d3d_skipped;
        }
        data.query_collect_cpu_ms=ms(started);
    }
    void discard()noexcept{for(auto& slot:slots)slot.pending=false;for(auto& slot:cuda_slots)slot.pending=false;active={};data={};}
    void release_cuda()noexcept{
        if(cu)for(auto& slot:cuda_slots)for(auto& event:slot.events)if(event){cu->cudaEventDestroy(event);event=nullptr;}
        cu=nullptr;stream=nullptr;event_failed=false;discard();
    }
    void attach(Cuda& cuda,cudaStream_t s){cu=&cuda;stream=s;}
    CudaSlot* reserve()noexcept{
        if(event_failed||!cu)return nullptr;
        for(auto& slot:cuda_slots){
            if(slot.pending)continue;
            if(!slot.events[0])for(auto& event:slot.events)if(cu->cudaEventCreateWithFlags(&event,cudaEventDefault)!=cudaSuccess){
                for(auto& created:slot.events)if(created){cu->cudaEventDestroy(created);created=nullptr;}
                event_failed=true;return nullptr;
            }
            slot.pending=true;slot.valid=true;return &slot;
        }
        return nullptr;
    }
    RifePhaseTiming* phase(float t,unsigned w,unsigned h){
        if(!data.enabled||data.phase_count>=data.phases.size())return nullptr;
        unsigned index=data.phase_count++;auto& sample=data.phases[index];
        sample.pair=data.pair;sample.pts=data.pts;sample.phase=t;sample.width=w;sample.height=h;
        active[index]=reserve();sample.cuda_pending=active[index]!=nullptr;
        if(!active[index])++data.cuda_skipped;return &sample;
    }
    CudaSlot* slot_for(RifePhaseTiming* sample)noexcept{return sample?active[size_t(sample-data.phases.data())]:nullptr;}
    void record(RifePhaseTiming* sample,unsigned i)noexcept{
        auto slot=slot_for(sample);if(slot&&slot->valid&&cu->cudaEventRecord(slot->events[i],stream)!=cudaSuccess)slot->valid=false;
    }
    void finish(RifePhaseTiming* sample)noexcept{
        if(auto slot=slot_for(sample)){slot->sample=*sample;if(read_cuda(*slot))*sample=slot->sample;}
    }
    Slot* begin_gpu(RifeGpuKind kind,float t,unsigned w,unsigned h){
        if(!data.enabled)return nullptr;
        for(auto& slot:slots){
            if(slot.pending)continue;
            if(!slot.disjoint){D3D11_QUERY_DESC desc{D3D11_QUERY_TIMESTAMP_DISJOINT,0};auto hr=d->CreateQuery(&desc,&slot.disjoint);
                desc.Query=D3D11_QUERY_TIMESTAMP;if(SUCCEEDED(hr))hr=d->CreateQuery(&desc,&slot.start);if(SUCCEEDED(hr))hr=d->CreateQuery(&desc,&slot.stop);
                if(FAILED(hr)){slot.disjoint.Reset();slot.start.Reset();slot.stop.Reset();++data.d3d_invalid;return nullptr;}}
            slot.sample={data.pair,data.pts,t,w,h,kind,0};slot.pending=true;c->Begin(slot.disjoint.Get());c->End(slot.start.Get());return &slot;
        }
        ++data.d3d_skipped;return nullptr;
    }
    void end_gpu(Slot* slot){if(slot){c->End(slot->stop.Get());c->End(slot->disjoint.Get());}}
};
