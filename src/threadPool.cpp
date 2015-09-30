#include <thread>
#include <atomic>
#include "threadPool.hpp"

#if defined(_WIN32) || defined(__linux)

namespace w2xc {

#if defined __i386__ || defined __x86_64__

#include <cpuid.h>
#define rmb() __asm__ __volatile__ ("":::"memory")
#define wmb() __asm__ __volatile__ ("":::"memory")

#endif

#ifdef _MSC_VER
#define rmb() _ReadBarrier()
#define wmb() _WriteBarrier()
#endif

#ifdef _WIN32

#include <process.h>

void
notify_event(HANDLE ev)
{
	wmb();
	SetEvent(ev);
}

void
wait_event(HANDLE ev)
{
	WaitForSingleObject(ev, INFINITE);
	rmb();
}

HANDLE
create_event()
{
	return CreateEvent(nullptr, FALSE, FALSE, nullptr);
}
void
delete_event(HANDLE h)
{
	CloseHandle(h);
}

#elif defined __linux

#include <unistd.h>
#include <sys/eventfd.h>

int
create_event()
{
	return eventfd(0,EFD_CLOEXEC);
}

void
notify_event(int fd)
{
	uint64_t ev_val = 1;
	wmb();
	ssize_t s = write(fd, &ev_val, sizeof(ev_val));
	if (s != sizeof(uint64_t)) {
		perror("write");    /* ?? */
	}
}

void
wait_event(int fd)
{
	uint64_t ev_val;

	ssize_t s = read(fd, &ev_val, sizeof(ev_val));
	if (s != sizeof(uint64_t)) {
		perror("read");    /* ?? */
	}

	rmb();
}

void
delete_event(int fd)
{
	close(fd);
}

#endif

void Thread::func() {
	while (1) {
		wait_event(to_client);
		rmb();

		if (this->p->fini_all) {
			return;
		}

		(*p->func)();

		int count = ++p->fini_count;
		if (count == p->num_thread) {
			notify_event(p->to_master);
		}
	}
}
void Thread::start(ThreadPool *p)
{
	this->p = p;
	wmb();
	t = std::thread(&Thread::func, this);
}

struct ThreadPool *
initThreadPool(int cpu)
{
	ThreadPool *ret = new ThreadPool;
	ret->to_master = create_event();
	ret->threads = new Thread[cpu];
	for (int i=0; i<cpu; i++) {
		ret->threads[i].start(ret);
	}
	ret->num_thread = cpu;
	ret->fini_all = false;
	return ret;
}

void
finiThreadPool(struct ThreadPool *p)
{
	p->fini_all = true;

	for (int i=0; i<p->num_thread; i++) {
		notify_event(p->threads[i].to_client);
	}
	for (int i=0; i<p->num_thread; i++) {
		p->threads[i].t.join();
	}

	delete [] p->threads;

	delete_event(p->to_master);
}

void
startFuncBody(struct ThreadPool *p, ThreadFuncBase *f)
{
	p->fini_count = 0;
	p->func = f;

	for (int i=0; i<p->num_thread; i++) {
		notify_event(p->threads[i].to_client);
	}

	wait_event(p->to_master);

	return;
}


}


#endif // __APPLE__
