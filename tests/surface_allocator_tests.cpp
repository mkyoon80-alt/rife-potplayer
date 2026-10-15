
#include "nvof/d3d11_transport.hpp"
#include <thread>
#include <iostream>
#include <stdexcept>
using Microsoft::WRL::ComPtr;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
 HRESULT hr=S_OK;ComPtr<nvof::transport::SurfaceAllocator> pool;pool=new nvof::transport::SurfaceAllocator(&hr,true);require(SUCCEEDED(hr),"allocator constructor");
 ALLOCATOR_PROPERTIES props{1,128,1,0},actual{};require(pool->SetProperties(&props,&actual)==S_OK&&pool->Commit()==S_OK,"commit");
 ComPtr<IMediaSample> sample;require(pool->GetBuffer(&sample,nullptr,nullptr,AM_GBF_NOWAIT)==S_OK,"initial sample");
 for(int i=0;i<10000;++i){IMediaSample* missing=reinterpret_cast<IMediaSample*>(1);require(pool->GetBuffer(&missing,nullptr,nullptr,AM_GBF_NOWAIT)==VFW_E_TIMEOUT&&missing==nullptr,"NOWAIT must report full");}
 require(!pool->has_spurious_wakeup_for_test(),"NOWAIT fabricated semaphore waiters");
 auto* surface=dynamic_cast<nvof::transport::SurfaceSample*>(sample.Get());require(surface!=nullptr,"surface type");surface->trace_submission(true);sample.Reset();
 require(!pool->has_spurious_wakeup_for_test(),"return released semaphore for nonwaiting callers");
 require(pool->GetBuffer(&sample,nullptr,nullptr,0)==S_OK,"reacquire");auto timing=surface->take_return_timing();require(timing.submitted!=nvof::transport::SurfaceSample::TimingClock::time_point{}&&timing.returned>=timing.submitted,"sample-owned return timing lost");
 surface->trace_submission(false);sample.Reset();pool->GetBuffer(&sample,nullptr,nullptr,0);timing=surface->take_return_timing();require(timing.returned==nvof::transport::SurfaceSample::TimingClock::time_point{},"unsampled cycle generated timestamps");
 HANDLE started=CreateEventW(nullptr,FALSE,FALSE,nullptr),done=CreateEventW(nullptr,FALSE,FALSE,nullptr);HRESULT result=E_UNEXPECTED;
 std::thread consumer([&]{SetEvent(started);ComPtr<IMediaSample> waited;result=pool->GetBuffer(&waited,nullptr,nullptr,0);SetEvent(done);});
 require(WaitForSingleObject(started,1000)==WAIT_OBJECT_0,"consumer start");bool blocked=WaitForSingleObject(done,20)==WAIT_TIMEOUT;sample.Reset();bool woke=WaitForSingleObject(done,1000)==WAIT_OBJECT_0;consumer.join();require(blocked&&woke&&result==S_OK,"blocking acquisition or return notification broken");
 pool->GetBuffer(&sample,nullptr,nullptr,0);result=E_UNEXPECTED;
 std::thread stopped([&]{SetEvent(started);ComPtr<IMediaSample> waited;result=pool->GetBuffer(&waited,nullptr,nullptr,0);SetEvent(done);});
 WaitForSingleObject(started,1000);blocked=WaitForSingleObject(done,20)==WAIT_TIMEOUT;pool->Decommit();woke=WaitForSingleObject(done,1000)==WAIT_OBJECT_0;stopped.join();require(blocked&&woke&&result==VFW_E_NOT_COMMITTED,"decommit did not abort allocation wait");sample.Reset();
 require(pool->GetBuffer(nullptr,nullptr,nullptr,AM_GBF_NOWAIT)==E_POINTER,"null output");pool->Commit();pool->GetBuffer(&sample,nullptr,nullptr,0);
 // Prove the regression test detects the legacy bug it guards against.
 IMediaSample* missing=nullptr;pool->CBaseAllocator::GetBuffer(&missing,nullptr,nullptr,AM_GBF_NOWAIT);require(pool->has_spurious_wakeup_for_test(),"legacy control failed to exhibit the bug");sample.Reset();pool->Decommit();CloseHandle(started);CloseHandle(done);
 std::cout<<"PASS 10000 NOWAIT misses/no false wakeups, return/reuse timing, blocking wake, decommit cancellation, legacy positive control\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
