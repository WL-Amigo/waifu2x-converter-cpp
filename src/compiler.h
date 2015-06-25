#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#define ALWAYS_INLINE __attribute__((always_inline))
#else
#define UNUSED
#define ALWAYS_INLINE __forceinline
#endif

