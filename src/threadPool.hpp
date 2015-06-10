#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

void initThreadPool(int cpu);
void finiThreadPool(void);

struct ThreadFuncBase {
    virtual void operator() () = 0;
    virtual ~ThreadFuncBase() { }
};

template<typename FuncT>
struct ThreadFunc
    :public ThreadFuncBase
{
    FuncT f;
    ThreadFunc(FuncT const &f)
        :f(f)
    {}

    virtual void operator()() {
        f();
    }

    virtual ~ThreadFunc(){}
};

extern void startFuncBody(ThreadFuncBase *f);

template <typename FuncT> void
startFunc(FuncT const &f)
{
    ThreadFuncBase *fb = new ThreadFunc<FuncT>(f);
    startFuncBody(fb);
    delete fb;
}

#endif