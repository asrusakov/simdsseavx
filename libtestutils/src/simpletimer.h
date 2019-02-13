/***************************
**
** simple timer for my purposes
**
*/
#pragma once
#include <chrono>
#include <stringstream>
#include <map>

namespace asr{

class SimpleTimer {

    //nreps -- optional argument. number of test repetetions. used for output normalization
    void start(const std::string &timernm="", long int nreps=1);
    void stop(const std::string &timernm="");
    void resume(const std::string &timernm="");

    std::string report(const std::string &timernm);        
    std::string reportall();
    
protected:

    typedef std::chrono::time_point<std::chrono::system_clock> timePoint;

    //store num of test repetions for normalization
    std::map<std::string, long int>  reps_;
    std::map<std::string, timePoint> starts_;
    std::map<std::string, timePoint> durations_;
}

    
void SimpleTimer::start(const std::string &timernm="", long int nreps=1){
    reps_[timernm] = nreps;
    auto &v = starts_[timernm];
    durations_[timernm] = 0;
    auto start = std::chrono::system_clock::now();
    v = start;
}

void SimpleTimer::stop(const std::string &timernm="") {
    auto end = std::chrono::system_clock::now();
    auto &v = storage_[timernm];
    durations_[timernm] = 0;

}

void SimpleTimer::resume(const std::string &timernm="") {
    auto &v =  starts_[timernm];
    auto start = std::chrono::system_clock::now();
    v = start;
}
