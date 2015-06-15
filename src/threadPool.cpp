#include <thread>
#include <atomic>
#include "threadPool.hpp"

#if defined __i386__ || defined __x86_64__
#include <cpuid.h>
#define rmb() __asm__ __volatile__ ("":::"memory")
#define wmb() __asm__ __volatile__ ("":::"memory")
#endif

#ifdef linux
#include <sys/eventfd.h>
#endif

#ifdef _WIN32

#include <windows.h>
#include <process.h>

typedef HANDLE event_t;

static void
notify_event(HANDLE ev)
{
	wmb();
	SetEvent(ev);
}

static void
wait_event(HANDLE ev)
{
	WaitForSingleObject(ev, INFINITE);
	rmb();
}

static HANDLE
create_event()
{
	return CreateEvent(nullptr, FALSE, FALSE, nullptr);
}
static void
delete_event(HANDLE h)
{
	CloseHandle(h);
}


#elif defined __linux
typedef int event_t;

#include <unistd.h>
#include <sys/eventfd.h>

static int
create_event()
{
	return eventfd(0,EFD_CLOEXEC);
}

static void
notify_event(int fd)
{
	uint64_t ev_val = 1;
	wmb();
	ssize_t s = write(fd, &ev_val, sizeof(ev_val));
	if (s != sizeof(uint64_t)) {
		perror("write");    /* ?? */
	}
}

static void
wait_event(int fd)
{
	uint64_t ev_val;

	ssize_t s = read(fd, &ev_val, sizeof(ev_val));
	if (s != sizeof(uint64_t)) {
		perror("read");    /* ?? */
	}

	rmb();
}

static void
delete_event(int fd)
{
	close(fd);
}

#endif

namespace {

struct Thread {
	event_t to_client;
	std::thread t;

	void func();
	Thread()
		:to_client(create_event()),
		 t(std::thread(&Thread::func, this))
	{}

	~Thread() {
		delete_event(to_client);
	}

	Thread(const Thread&) = delete;
	Thread&  operator=(const Thread&) = delete;
};


struct ThreadPool {
	int num_thread;
	std::atomic<int> fini_count;
	bool fini_all;

	Thread *threads;
	event_t to_master;
	ThreadFuncBase *func;
};

static ThreadPool threadPool;

void Thread::func() {
	while (1) {
		wait_event(to_client);
		rmb();

		if (threadPool.fini_all) {
			return;
		}

		(*threadPool.func)();

		int count = ++threadPool.fini_count;
		if (count == threadPool.num_thread) {
			notify_event(threadPool.to_master);
		}
	}
}

}

void
initThreadPool(int cpu)
{
	threadPool.to_master = create_event();
	threadPool.threads = new Thread[cpu];
	threadPool.num_thread = cpu;
	threadPool.fini_all = false;
}

void
finiThreadPool()
{
	threadPool.fini_all = true;

	for (int i=0; i<threadPool.num_thread; i++) {
		notify_event(threadPool.threads[i].to_client);
	}
	for (int i=0; i<threadPool.num_thread; i++) {
		threadPool.threads[i].t.join();
	}

	delete [] threadPool.threads;

	delete_event(threadPool.to_master);
}

void
startFuncBody(ThreadFuncBase *f)
{
	threadPool.fini_count = 0;
	threadPool.func = f;
	wmb();

	for (int i=0; i<threadPool.num_thread; i++) {
		notify_event(threadPool.threads[i].to_client);
	}

	wait_event(threadPool.to_master);
	rmb();

	return;
}
