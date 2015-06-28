#ifndef W2XC_SEC_HPP
#define W2XC_SEC_HPP

#include "compiler.h"

#ifdef _WIN32
#include <windows.h>

static double UNUSED
getsec(void)
{
    LARGE_INTEGER c;
    LARGE_INTEGER freq;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);

    return c.QuadPart/ (double)freq.QuadPart;
}


#elif __MACH__
#include <sys/time.h>
static int UNUSED
clock_gettime(int /* clk_id*/, struct timespec* t) {
	struct timeval now;
	int rv = gettimeofday(&now, NULL);
	if (rv) return rv;
	t->tv_sec  = now.tv_sec;
	t->tv_nsec = now.tv_usec * 1000;
	return 0;
}

static double // UNUSED
getsec(void)
{
    struct timespec ts;
    clock_gettime(0, &ts);

    return (ts.tv_sec) + (ts.tv_nsec / (1000.0*1000.0*1000.0));
}

#else

#include <time.h>
#include <unistd.h>

static double UNUSED
getsec(void)
{
    struct timespec ts;

#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif

    return (ts.tv_sec) + (ts.tv_nsec / (1000.0*1000.0*1000.0));
}

#endif

#endif