/*
* The MIT License (MIT)
* Copyright (c) 2015 amigo(white luckers), tanakamura, DeadSix27, YukihoAA and contributors
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

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

namespace w2xc
{
	void notify_event(event_t ev);
	event_t create_event(void);
	void delete_event(event_t ev);
	void notify_event(event_t ev);

	struct ThreadPool;

	struct ThreadFuncBase
	{
		virtual void operator() () = 0;
		virtual ~ThreadFuncBase() { }
	};
	template<typename FuncT>
	struct ThreadFunc : public ThreadFuncBase
	{
		FuncT f;
		ThreadFunc(FuncT const &f) : f(f) {}

		virtual void operator()()
		{
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

	struct Thread
	{
		ThreadPool *p;
		event_t to_client;
		std::thread t;

		void func();
		Thread() : to_client(create_event()) {}

		void start(ThreadPool *p);

		~Thread()
		{
			delete_event(to_client);
		}

		Thread(const Thread&) = delete;
		Thread&	operator=(const Thread&) = delete;
		Thread&	operator=(Thread&&) = delete;
	};

	struct ThreadPool
	{
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
