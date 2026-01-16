#include "rmsynth.h"
#include "../../core/rm_code.h"

extern "C" {

const char* rmsynth_version(void) {
    return "0.0.1";
}

int rmsynth_rm_dimension(int n, int r) {
    return rm_dimension(n, r);
}

}
