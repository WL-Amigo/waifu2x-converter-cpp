#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <Shlwapi.h>
#include <windowsx.h>
#include <process.h>
#include "w2xconv.h"
#include "w2xcr.h"

static int block_size = 0;

enum param_denoise_code {
    DENOISE_NONE,
    DENOISE_0,
    DENOISE_1,
    DENOISE_2,
    DENOISE_JPEG_0,
    DENOISE_JPEG_1,
    DENOISE_JPEG_2,
};

static int param_denoise = DENOISE_1;
static double param_scale = 2.0;

#define WIN_WIDTH 600
#define DISP_HEIGHT 60

enum packet_type {
    PKT_START,
    PKT_FINI_REQUEST,
    PKT_FINI_ALL,
    PKT_PROGRESS,
};

struct packet {
    enum packet_type tp;

    union {
        struct {
            int dev;
        } progress;

        struct {
            int dev;
        } start;
    }u;
};

#define QUEUE_SIZE 16

struct Queue {
    struct packet packets[QUEUE_SIZE];

    int rd;
    int wr;

    HANDLE ev;
    CRITICAL_SECTION cs;
};

static int
recv_packet(struct Queue *q, struct packet *p)
{
    int ret = -1;

    EnterCriticalSection(&q->cs);
    if (q->rd == q->wr) {
        ret = -1;
        ResetEvent(q->ev);
    } else {
        ret = 0;
        *p = q->packets[q->rd];
        q->rd = (q->rd + 1)%QUEUE_SIZE;
    }
    LeaveCriticalSection(&q->cs);

    return ret;
}

static int
send_packet(struct Queue *q, struct packet *p)
{
    EnterCriticalSection(&q->cs);
    q->packets[q->wr] = *p;
    q->wr = (q->wr+1)%QUEUE_SIZE;
    SetEvent(q->ev);
    LeaveCriticalSection(&q->cs);

    return 0;
}

static void
queue_init(struct Queue *q)
{
    q->rd = 0;
    q->wr = 0;
    q->ev = CreateEvent(NULL, TRUE, FALSE, NULL);
    InitializeCriticalSection(&q->cs);
}

enum app_state {
    STATE_RUN,
    STATE_FINI_REQUEST,
    STATE_FINI
};

struct path_pair {
    int denoise;

    wchar_t *src_path;
    wchar_t *dst_path;
};

struct app {
    CRITICAL_SECTION app_cs;
    enum app_state state;
    HWND win;

    int processed;

    struct Queue from_worker;

    int num_thread;
    struct thread_arg *threads;

    int path_list_capacity;
    int path_list_nelem;
    struct path_pair *path_list;

    int *dev_list;
    const struct W2XConvProcessor *proc_list;

    int cursor;
};

static int
add_file(struct app *app, const wchar_t *src_path)
{
    int r;
    wchar_t *dst_path;
    size_t path_len;

    wchar_t drive[_MAX_DRIVE];
    wchar_t dir[_MAX_DIR];
    wchar_t fname[_MAX_FNAME];
    wchar_t ext[_MAX_EXT];

    WIN32_FIND_DATAW orig_st, mai_st;
    HANDLE finder;
    int denoise = -1;

    int denoise_jpg = -1;
    int denoise_default = -1;

    switch ((enum param_denoise_code)param_denoise) {
    case DENOISE_NONE:
        denoise_jpg = -1;
        denoise_default = -1;
        break;
    case DENOISE_0:
        denoise_jpg = 0;
        denoise_default = 0;
        break;
    case DENOISE_1:
        denoise_jpg = 1;
        denoise_default = 1;
        break;
    case DENOISE_2:
        denoise_jpg = 2;
        denoise_default = 2;
        break;

    case DENOISE_JPEG_0:
        denoise_jpg = 0;
        denoise_default = -1;
        break;
    case DENOISE_JPEG_1:
        denoise_jpg = 1;
        denoise_default = -1;
        break;
    case DENOISE_JPEG_2:
        denoise_jpg = 2;
        denoise_default = -1;
        break;
    }

    _wsplitpath(src_path,
               drive,
               dir,
               fname,
               ext);

     {
        char header[8];
        FILE *fp = _wfopen(src_path, L"rb");
        if (fp == NULL) {
            return 0;
        }
        size_t sz = fread(header, 1, 8, fp);
        fclose(fp);
        if (sz != 8) {
            return 0;
        }

        const static char jpg[] = {0xFF, 0xD8, 0xff};
        const static char png[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
        const static char bmp[] = {0x42, 0x4D};

        if (memcmp(header,jpg,3) == 0) {
            denoise = denoise_jpg;
        } else if (memcmp(header,png,8) == 0 ||
                   memcmp(header,bmp,2) == 0)
        {
            denoise = denoise_default;
        } else {
            return 0;
        }
     }

    if (wcsncmp(fname, L"mai_", 4) == 0) {
        return 0;
    }

    path_len = wcslen(src_path);
    size_t dst_path_len = path_len + 5 + 4;
    dst_path = malloc(dst_path_len * sizeof(wchar_t));

    r = _snwprintf(dst_path, dst_path_len,
                  L"%s%smai_%s.png",
                  drive, dir, fname);
    if (r > dst_path_len) {
        free(dst_path);
        return 0;
    }

    finder = FindFirstFileW(src_path, &orig_st);
    if (finder == INVALID_HANDLE_VALUE) {
        free(dst_path);
        return 0;
    }
    FindClose(finder);

    finder = FindFirstFileW(dst_path, &mai_st);
    FindClose(finder);

    if (finder != INVALID_HANDLE_VALUE) {
        uint64_t orig_time = (((uint64_t)orig_st.ftLastWriteTime.dwHighDateTime)<<32) |
            ((uint64_t)orig_st.ftLastWriteTime.dwLowDateTime);
        uint64_t mai_time = (((uint64_t)mai_st.ftLastWriteTime.dwHighDateTime)<<32) |
            ((uint64_t)mai_st.ftLastWriteTime.dwLowDateTime);

        if (mai_time >= orig_time) {
            //printf("newer : %s > %s\n", dst_path, src_path);
            free(dst_path);
            return 0;
        }
    }

    if (app->path_list_nelem == app->path_list_capacity) {
        app->path_list_capacity *= 2;
        app->path_list = realloc(app->path_list, app->path_list_capacity * sizeof(struct path_pair));
    }

    app->path_list[app->path_list_nelem].denoise = denoise;
    app->path_list[app->path_list_nelem].src_path = wcsdup(src_path);
    app->path_list[app->path_list_nelem].dst_path = dst_path;

    app->path_list_nelem++;

    return r;
}

static int
traverse_dir(struct app *app, const wchar_t *src_path)
{
    WIN32_FIND_DATAW fd;
    HANDLE finder;

    wchar_t pattern[MAX_PATH] = L"";
    wchar_t sub[MAX_PATH] = L"";
    int r = 0;

    PathCombineW(pattern, src_path, L"*");

    finder = FindFirstFileW(pattern, &fd);
    if (finder != INVALID_HANDLE_VALUE) {
        do {
            if ((wcscmp(fd.cFileName,L".") == 0) ||
                (wcscmp(fd.cFileName,L"..") == 0))
            {
                /* ignore */
            } else {
                PathCombineW(sub, src_path, fd.cFileName);

                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    r = traverse_dir(app, sub);
                } else {
                    r = add_file(app, sub);
                }

                if (r < 0) {
                    break;
                }
            }
        } while(FindNextFileW(finder, &fd));

        FindClose(finder);
    }

    return r;
}

struct thread_arg {
    int dev_id;
    struct app *app;
    HANDLE thread_handle;

    wchar_t *cur_path;
    double cur_flops;
};

static unsigned int __stdcall
proc_thread(void *ap)
{
    struct thread_arg *ta = (struct thread_arg*)ap;
    struct app *app = ta->app;

    struct W2XConv *c;
    int r;
    int ret = 1;
    size_t path_len = 4;
    WCHAR *self_path = (WCHAR*)malloc((path_len+1) * sizeof(WCHAR));
    DWORD len;

    c = w2xconv_init_with_processor(ta->dev_id, 0, 0);
    while (1) {
        len = GetModuleFileNameW(NULL, self_path, (DWORD) path_len);
        if (len > 0 && len != path_len) {
            break;
        }

        path_len *= 2;
        self_path = (WCHAR*)realloc(self_path, (path_len+1) * sizeof(WCHAR));
    }

    {
        size_t cur;
        for (cur=wcslen(self_path); cur>=0; cur--) {
            if (self_path[cur] == L'\\') {
                self_path[cur] = L'\0';
                break;
            }
        }

        WCHAR *models_path = malloc((wcslen(self_path) + 11 + 1) * sizeof(WCHAR));

        wsprintf(models_path, L"%s\\models_rgb", self_path);
		// You should not have Unicode char in path where model files located.
        r = w2xconv_load_models(c, models_path);
        free(self_path);
        free(models_path);

        if (r < 0) {
            goto error;
        }
    }


    {
        struct packet pkt;
        pkt.tp = PKT_START;
        pkt.u.start.dev = ta->dev_id;
        send_packet(&app->from_worker, &pkt);
    }

    while (1) {
        struct packet pkt;
        struct path_pair *p;
        int li;
        EnterCriticalSection(&app->app_cs);
        li = app->cursor;
        if (li < app->path_list_nelem) {
            app->cursor++;
        }
        LeaveCriticalSection(&app->app_cs);

        if (li >= app->path_list_nelem) {
            /* finished */
            break;
        }
		
		int imwrite_params[6];
		imwrite_params[0] = 64; //cv::IMWRITE_WEBP_QUALITY;
		imwrite_params[1] = 90;
		imwrite_params[2] = 1;  //cv::IMWRITE_JPEG_QUALITY;
		imwrite_params[3] = 90;
		imwrite_params[4] = 16; //cv::IMWRITE_PNG_COMPRESSION
		imwrite_params[5] = 5;

        p = &app->path_list[li];
        r = w2xconv_convert_file(c,
                                 p->dst_path,
                                 p->src_path,
                                 p->denoise,
                                 param_scale, block_size, imwrite_params);

        if (r != 0) {
            goto error;
        }

        pkt.tp = PKT_PROGRESS;
        pkt.u.progress.dev = ta->dev_id;
        send_packet(&app->from_worker, &pkt);

        ta->cur_path = p->src_path;
        ta->cur_flops = c->flops.flop / c->flops.process_sec;

        _ReadWriteBarrier();
        if (app->state == STATE_FINI_REQUEST) {
            goto fini;
        }
    }

    ret = 0;
    goto fini;

error:
    if (r != -2) {
        char *err = w2xconv_strerror(&c->last_error);
        MessageBoxA(NULL, err, "error", MB_OK);
        w2xconv_free(err);
    }

fini:
    w2xconv_fini(c);
    {
        struct packet pkt;
        pkt.tp = PKT_FINI_ALL;
        send_packet(&app->from_worker, &pkt);
    }

    return ret;
}

static void
on_close(HWND wnd)
{
    struct app *app = (struct app*)GetWindowLongPtr(wnd, GWLP_USERDATA);
    PostQuitMessage(0);

    {
        app->state = STATE_FINI_REQUEST;
        _WriteBarrier();
    }

}

static BOOL
on_create(HWND wnd, LPCREATESTRUCT cp)
{
    struct app *app = cp->lpCreateParams;
    unsigned threadID;

    int dev_list_size = 0, i;
    int dev_list_cap = 2;
    int *dev_list = malloc(sizeof(int) * dev_list_cap);

    const struct W2XConvProcessor *proc_list;
    size_t num_all_dev;

    SetWindowLongPtr(wnd, GWLP_USERDATA, (LONG_PTR)app);
    proc_list = w2xconv_get_processor_list(&num_all_dev);

    for (i=0; i<(int) num_all_dev; i++) {
        const struct W2XConvProcessor *p = &proc_list[i];

        if ((p->type == W2XCONV_PROC_HOST) ||
            (p->type == W2XCONV_PROC_CUDA) ||
            (p->type == W2XCONV_PROC_OPENCL && ((p->sub_type == W2XCONV_PROC_OPENCL_AMD_GPU) ||
                                                (p->sub_type == W2XCONV_PROC_OPENCL_INTEL_GPU)))
            )
        {
            if (dev_list_cap == dev_list_size) {
                dev_list_cap *= 2;
                dev_list = realloc(dev_list, sizeof(int) * dev_list_cap);
            }

            dev_list[dev_list_size++] = i;
        }
    }

    if (app->path_list_nelem < dev_list_size) {
        dev_list_size = app->path_list_nelem;
    }

    if (dev_list_size == 0) {
        dev_list_size = 1;
    }

    struct thread_arg *threads = malloc(sizeof(struct thread_arg) * dev_list_size);

    for (i=0; i<dev_list_size; i++) {
        threads[i].dev_id = dev_list[i];
        threads[i].app = app;
        threads[i].cur_flops = 0;
        threads[i].cur_path = NULL;
        threads[i].thread_handle = (HANDLE)(uintptr_t)_beginthreadex(NULL, 0, proc_thread, &threads[i], 0, &threadID);
    }

    app->num_thread = dev_list_size;
    app->threads = threads;
    app->dev_list = dev_list;
    app->proc_list = proc_list;


    {
        RECT rect;
        rect.left = 0;
        rect.top = 0;
        rect.right = WIN_WIDTH;
        rect.bottom = DISP_HEIGHT * (dev_list_size+1);

        AdjustWindowRectEx( &rect, WS_OVERLAPPEDWINDOW, FALSE, 0 );
        SetWindowPos(wnd, NULL, 0, 0,
                     rect.right-rect.left,
                     rect.bottom-rect.top,
                     SWP_NOMOVE|SWP_NOZORDER);
    }

    return TRUE;
}

static LRESULT CALLBACK
wnd_proc(HWND hwnd,
         UINT uMsg,
         WPARAM wParam,
         LPARAM lParam)
{
    switch (uMsg) {
        HANDLE_MSG(hwnd, WM_CLOSE, on_close);
        HANDLE_MSG(hwnd, WM_CREATE, on_create);

    default:
        break;
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

static void
update_display(struct app *app)
{
    HDC dc = GetDC(app->win);
    RECT r;

    GetClientRect(app->win, &r);
    FillRect(dc, &r, GetStockBrush(WHITE_BRUSH));

    if ((app->processed == 0) && app->state == STATE_FINI) {
        static const wchar_t msg[] = L"nop";
        TextOutW(dc, 10, 10, msg, sizeof(msg)-1);
    } else {
        wchar_t line[4096];
        wchar_t path[128];
        int i;

        if (app->processed != 0) {
            _snwprintf(line, sizeof(line),
                      L"progress : %d/%d",
                      app->processed,
                      app->path_list_nelem);
        }

        TextOutW(dc, 10, 10, line, (int) wcslen(line));

        for (i=0; i<app->num_thread; i++) {
            wchar_t *cur_path = app->threads[i].cur_path;
            double cur_flops = app->threads[i].cur_flops;


            if (cur_path) {
                size_t len = wcslen(cur_path), line_len;
                //HGDIOBJ old_brush;
                const struct W2XConvProcessor *proc;

                if (len > 30) {
                    size_t rem = len - 30;
                    _snwprintf(path, 128, L"...%s", cur_path + rem);
                } else {
                    wcscpy(path, cur_path);
                }

                if (app->state == STATE_FINI) {
                    _snwprintf(line, sizeof(line),
                              L"%.2f[GFLOPS] %s [Complete!!]",
                              cur_flops/(1000.0*1000.0*1000.0),
                              path);
                } else {
                    _snwprintf(line, sizeof(line),
                              L"%.2f[GFLOPS] %s",
                              cur_flops/(1000*1000*1000.0),
                              path);
                }

                line_len = wcslen(line);

                proc = &app->proc_list[app->dev_list[i]];

                TextOutA(dc, 10, DISP_HEIGHT*(i+1) + 10, proc->dev_name, (int) strlen(proc->dev_name));
                TextOutW(dc, 10, DISP_HEIGHT*(i+1) + 28, line, (int) line_len);
            }
        }
    }
    ReleaseDC(app->win, dc);
}

static INT_PTR CALLBACK
initdlg_callback(HWND wnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    wchar_t buf[256];
	wchar_t* end;
    switch (message) {
    case WM_CLOSE:
        EndDialog(wnd, -1);
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case IDC_OK:
            param_denoise = (int) SendDlgItemMessageW(wnd, IDC_DENOISE, CB_GETCURSEL, 0, 0);
            GetDlgItemTextW(wnd, IDC_SCALE, buf, 256);
            param_scale = wcstof(buf, &end);
            if (param_scale != 0) {
                EndDialog(wnd, 0);
            } else {
                MessageBoxW(NULL, L"invalid scale value", L"w2xcr", MB_OK);
            }
            return (INT_PTR)TRUE;

        case IDC_CANCEL:
            EndDialog(wnd, -1);
            return (INT_PTR)TRUE;
        }
        break;

    case WM_INITDIALOG:
        SetDlgItemTextW(wnd, IDC_SCALE, L"2.0");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"none");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"0");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"1");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"2");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"0(jpeg only)");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"1(jpeg only)");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_ADDSTRING, 0, (LPARAM)L"2(jpeg only)");
        SendDlgItemMessageW(wnd, IDC_DENOISE, CB_SETCURSEL, 1, 0);
        break;
    }

    return (INT_PTR)FALSE;
}

#define WINMAIN

#ifdef WINMAIN
#pragma comment ( linker, "/entry:\"wWinMainCRTStartup\"" )
int WINAPI
wWinMain(HINSTANCE hInst,
        HINSTANCE hPrev,
        LPWSTR lpCmdLine,
        int nCmdShow)
#else
#pragma comment ( linker, "/entry:\"wmainCRTStartup\"" )
int wmain(int argc, wchar_t **argv)
#endif
{
#ifdef WINMAIN
    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
#else
    HINSTANCE hInst = GetModuleHandle(NULL);
#endif

    WNDCLASSEXW cls={0};
    struct app app;
    HWND win;
    int run = 1, fini_count;
    int argi, i;
    DWORD pathlen, attr;
    wchar_t *fullpath, *filepart;
    wchar_t *src_path;
    int interactive = 0;

    for (argi=1; argi < argc; argi++) {
        if (wcscmp(argv[argi],L"--block_size") == 0) {
            block_size = wcstol(argv[argi+1], NULL, 10);
            argi++;
        } else if (wcscmp(argv[argi], L"--interactive") == 0) {
            interactive = 1;
        } else {
            break;
        }
    }

    if (interactive) {
        INT_PTR r = DialogBox(hInst, MAKEINTRESOURCE(IDD_INIT), NULL, initdlg_callback);
        if (r == -1) {
            return 1;
        }
    }

    int rem = argc - argi;
    if (rem == 0) {
        MessageBoxW(NULL, L"usage : w2xcr [--block_size <size>] <in>", L"w2xcr", MB_OK);
        return 1;
    }

    app.path_list_capacity = 4;
    app.path_list_nelem = 0;
    app.path_list = malloc(sizeof(struct path_pair) * app.path_list_capacity);


    for (; argi<argc; argi++) {
        src_path = argv[argi];

        pathlen = GetFullPathNameW(src_path, 0, NULL, NULL);
        if (pathlen == 0) {
            MessageBoxW(NULL, L"open failed", src_path, MB_OK);
            return 1;
        }

        fullpath = malloc(pathlen * sizeof(wchar_t));
        GetFullPathNameW(src_path, pathlen, fullpath, &filepart);

        attr = GetFileAttributesW(fullpath);
        if (attr < 0) {
            MessageBoxW(NULL, L"open failed", fullpath, MB_OK);
            return 1;
        }

        if (attr & FILE_ATTRIBUTE_DIRECTORY) {
            traverse_dir(&app, fullpath);
        } else {
            add_file(&app, fullpath);
        }

        free(fullpath);
    }

    cls.cbSize = sizeof(cls);
    cls.style = CS_CLASSDC;
    cls.lpfnWndProc = wnd_proc;
    cls.hInstance = hInst;
    cls.hIcon = NULL;
    cls.hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW);
    cls.hbrBackground = NULL;
    cls.lpszMenuName = NULL;
    cls.lpszClassName = L"w2xcr";
    cls.hIconSm = NULL;

    RegisterClassExW(&cls);

    queue_init(&app.from_worker);

    win = CreateWindowExW(0,
                         L"w2xcr",
                         L"w2xcr",
                         WS_OVERLAPPEDWINDOW,
                         CW_USEDEFAULT,
                         CW_USEDEFAULT,
                         WIN_WIDTH,
                         DISP_HEIGHT,
                         NULL,
                         NULL,
                         hInst,
                         &app);

    app.win = win;

    ShowWindow(win, SW_SHOW);
    UpdateWindow(win);

    app.state = STATE_RUN;
    app.processed = 0;

    InitializeCriticalSection(&app.app_cs);
    app.cursor = 0;

    fini_count = 0;

    while (run) {
        HANDLE list[1] = {app.from_worker.ev};

        DWORD r = MsgWaitForMultipleObjects(1, list, FALSE, INFINITE, QS_ALLEVENTS);

        if (r == WAIT_OBJECT_0) {
            struct packet pkt;

            while (1) {
                int r = recv_packet(&app.from_worker, &pkt);
                if (r < 0) {
                    break;
                }

                switch (pkt.tp) {
                case PKT_START:
                    break;

                case PKT_PROGRESS:
                    app.processed++;
                    update_display(&app);

                    break;

                case PKT_FINI_ALL: {
                    fini_count++;
                    if (fini_count == app.num_thread) {
                        app.state = STATE_FINI;
                    }
                    update_display(&app);
                }
                    break;

                case PKT_FINI_REQUEST:
                    /* ?? */
                    break;
                }
            }
        } else if (r == WAIT_OBJECT_0+1) {
            MSG msg;

            while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    run = 0;
                } else {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
            }
        }
    }

    for (i=0; i<app.num_thread; i++) {
        struct thread_arg *t = &app.threads[i];
        WaitForSingleObject(t->thread_handle, INFINITE);
        CloseHandle(t->thread_handle);
    }

    free(app.dev_list);

    return 0;
}
