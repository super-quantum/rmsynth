#ifndef RMSYNTH_H
#define RMSYNTH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


#define RMSYNTH_VERSION_MAJOR 0
#define RMSYNTH_VERSION_MINOR 0
#define RMSYNTH_VERSION_PATCH 1


const char* rmsynth_version(void);


int rmsynth_rm_dimension(int n, int r);


#ifdef __cplusplus
}
#endif

#endif
