#ifndef RM_SYNTH_PARALLEL_H
#define RM_SYNTH_PARALLEL_H

#pragma once
#include <cstddef>
#include <cstdlib>

// Global switch for parallel execution.
//
// If RM_PARALLEL is unset, parallel execution is enabled by default wherever the library was compiled with TBB or OpenMP support.
//
// If RM_PARALLEL is set to a string beginning with '0', 'f', 'F', 'n', or 'N', bparallel execution is disabled and code runs single-threaded.
inline bool rm_parallel_enabled() {
    const char* v = std::getenv("RM_PARALLEL");
    if (!v) return true;                       // default: parallel ON if compiled
    if (v[0]=='0' || v[0]=='f' || v[0]=='F' || v[0]=='n' || v[0]=='N') return false;
    return true;
}

#if defined(RM_HAVE_TBB)

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

// TBB-based parallel for wrapper.
//
// Calls f(i) for i in [begin, end).  If parallel execution is disabled or the range is too small, it falls back to a simple serial loop.
template <typename F>
inline void rm_par_for(std::size_t begin, std::size_t end, F f) {
    if (!rm_parallel_enabled() || end <= begin + 1) {
        for (std::size_t i=begin;i<end;++i) f(i);
        return;
    }
    tbb::parallel_for(tbb::blocked_range<std::size_t>(begin,end),
                      [&](const tbb::blocked_range<std::size_t>& r){
                          for (std::size_t i=r.begin(); i<r.end(); ++i) f(i);
                      });
}

#elif defined(RM_HAVE_OPENMP)

#include <omp.h>

// OpenMP-based parallel for wrapper.
template <typename F>
inline void rm_par_for(std::size_t begin, std::size_t end, F f) {
    if (!rm_parallel_enabled() || end <= begin + 1) {
        for (std::size_t i=begin;i<end;++i) f(i);
        return;
    }
#pragma omp parallel for schedule(dynamic)
    for (long long i=(long long)begin; i<(long long)end; ++i) {
        f((std::size_t)i);
    }
}

#else

// Fallback: no parallel backend, always run serially.
template <typename F>
inline void rm_par_for(std::size_t begin, std::size_t end, F f) {
    for (std::size_t i=begin;i<end;++i) f(i);
}

#endif

#endif //RM_SYNTH_PARALLEL_H
