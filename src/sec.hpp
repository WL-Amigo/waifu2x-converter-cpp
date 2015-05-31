#ifdef _WIN32
#include <windows.h>

static double
getsec(void)
{
    LARGE_INTEGER c;
    LARGE_INTEGER freq;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);

    return c.QuadPart/ (double)freq.QuadPart;
}

#else

#include <time.h>
#include <unistd.h>

static double
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