#include "nvof/startup_original_window.hpp"
#include <iostream>
#include <stdexcept>
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    nvof::StartupOriginalWindow w;
    require(!w.decide(1,0).original,"disabled by default");
    for(int rate:{24,25,30,60}){
        w.reset();w.set_rate(1);
        require(w.decide(1000,0).original,"first picture is immediate before Run");
        require(w.decide(9000,0).original,"paused graph must not consume window");
        require(w.run(10000),"first Run arms");
        require(!w.run(10020),"resume must not rearm");
        for(int i=1;i<rate;++i){
            const auto d=w.decide(10001,10000000LL*i/rate);
            if(10000000LL*i/rate<1000000)require(d.original&&!d.wait_ms,"source originals below boundary");
            else {require(!d.original&&d.wait_ms==99,"burst bounded without early inference");break;}
        }
        require(!w.decide(10100,2000000).original&&!w.active(),"actual 100-ms deadline");
        require(!w.run(50000),"later pause does not rearm");
    }
    w.reset();w.run(0);require(w.decide(99,0).original,"99-ms original");
    require(!w.decide(100,416667).original,"late decoder resumes immediately");
    w.reset();w.decide(0,0);w.run(10);require(w.decide(20,2000000).wait_ms==90,"wait remaining only");
    w.cancel();require(!w.decide(21,2000000).wait_ms&&!w.run(22),"seek cancels without rearm");
    w.reset();require(!w.decide(0,-1).original,"negative preroll rejected");
    w.decide(0,10000000);w.run(10);require(!w.decide(20,0).original&&!w.active(),"backward timestamp cancels");
    w.reset();w.set_rate(2);w.decide(0,0);w.run(10);
    require(w.decide(20,1500000).original&&w.decide(20,2000000).wait_ms==90,"rate scales media cap, not wall duration");
    w.reset();w.set_rate(1e300);w.decide(0,0);w.run(0);
    require(w.decide(1,1000000).original,"large rate does not overflow");
    w.reset();w.set_rate(1e-300);w.decide(0,0);w.run(0);
    require(w.decide(1,1).wait_ms==99,"small rate retains positive media bound");
    for(int rate:{24,60}){
        w.reset();w.set_rate(1);w.decide(0,0);w.run(1000);
        const int64_t boundary=((rate+9)/10)*10000000LL/rate;
        require(!w.begin_preparation(1010,boundary-10000000LL/rate),"no preparation during originals");
        require(w.begin_preparation(1020,boundary),"boundary may prepare early");
        require(!w.decide(1030,boundary+10000000LL/rate).original,"next endpoint can prepare");
        require(w.output_wait_ms(1060)==40,"ready phase still withheld until deadline");
        require(w.output_wait_ms(1099)==1,"no early release");
        require(w.output_wait_ms(1100)==0&&!w.active(),"same deadline, no extension");
        require(!w.run(2000),"no new window on resume");
        w.reset();w.decide(0,0);w.run(1000);w.begin_preparation(1020,boundary);w.cancel();
        require(!w.output_wait_ms(1021),"flush cancels prepared output wait");
    }
    std::cout<<"PASS startup originals: wall deadline, burst bound, first picture, 24/25/30/60p, pause, seek cancellation, rate\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
