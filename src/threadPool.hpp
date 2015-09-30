#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#if defined(_WIN32) || defined(__linux)

#include <thread>
#include <atomic>

#ifdef __linux

typedef int event_t;
#endif

#ifdef _WIN32

#include <windows.h>

typedef HANDLE event_t;

#endif



namespace w2xc {

void notify_event(event_t ev);
event_t create_event(void);
void delete_event(event_t ev);
void notify_event(event_t ev);

struct ThreadPool;

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

extern void startFuncBody(ThreadPool *p, ThreadFuncBase *f);

template <typename FuncT> void
startFunc(ThreadPool *p, FuncT const &f)
{
	ThreadFuncBase *fb = new ThreadFunc<FuncT>(f);
	startFuncBody(p, fb);
	delete fb;
}

struct Thread {
	ThreadPool *p;
	event_t to_client;
	std::thread t;

	void func();
	Thread()
		:to_client(create_event())
	{}

	void start(ThreadPool *p);

	~Thread() {
		delete_event(to_client);
	}

	Thread(const Thread&) = delete;
	Thread&	 operator=(const Thread&) = delete;
	Thread&	 operator=(Thread&&) = delete;
};


struct ThreadPool {
	int num_thread;
	std::atomic<int> fini_count;
	std::atomic<bool> fini_all;

	Thread *threads;
	event_t to_master;
	ThreadFuncBase *func;
};


struct ThreadPool * initThreadPool(int cpu);
void finiThreadPool(struct ThreadPool *p);

}

#endif // __APPLE__

#endif
