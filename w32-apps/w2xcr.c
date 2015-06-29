#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <Shlwapi.h>
#include <windowsx.h>
#include "w2xconv.h"

#define WIN_WIDTH 600
#define WIN_HEIGHT 100

enum packet_type {
    PKT_START,
    PKT_END_SUCCESS,
    PKT_END_FAILED,
    PKT_FINI_REQUEST,
    PKT_FINI_ALL,
    PKT_FINI_ERROR,
    PKT_PROGRESS,
};

struct packet {
    enum packet_type tp;

    union {
        struct {
            char *path;
            double flops;
        } progress;

        struct {
            char *dev_name;
        } start;
    }u;
};

#define QUEUE_SIZE 16

struct Queue {
    struct packet packets[QUEUE_SIZE];

    int rd;
    int wr;

    HANDLE ev;
};

static int
recv_packet(struct Queue *q, struct packet *p, int wait)
{
    while (1) {
        if (q->rd == q->wr) {
            if(wait) {
                WaitForSingleObject(q->ev, INFINITE);
            } else {
                DWORD r = WaitForSingleObject(q->ev, 0);
                if (r == WAIT_TIMEOUT) {
                    return -1;
                }
            }
            ResetEvent(q->ev);
        } else {
            *p = q->packets[q->rd];
            break;
        }
    }

    q->rd = (q->rd + 1)%QUEUE_SIZE;

    return 0;
}

static int
send_packet(struct Queue *q, struct packet *p)
{
    q->packets[q->wr] = *p;
    q->wr = (q->wr+1)%QUEUE_SIZE;

    SetEvent(q->ev);

    return 0;
}

static void
queue_init(struct Queue *q)
{
    q->rd = 0;
    q->wr = 0;
    q->ev = CreateEvent(NULL, TRUE, FALSE, NULL);
}

enum app_state {
    STATE_RUN,
    STATE_FINI
};

struct app {
    enum app_state state;
    HWND win;

    char *src_path;
    char *dev_name;

    HANDLE worker_thread;

    struct Queue to_worker;
    struct Queue from_worker;

    char *cur_path;
    double cur_flops;
};


static int
run1(struct app *app, struct W2XConv *c, const char *src_path)
{
    int r;
    char *dst_path;
    size_t path_len;

    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    WIN32_FIND_DATA orig_st, mai_st;
    HANDLE finder;
    struct packet pkt;

    {
        DWORD r = WaitForSingleObject(app->to_worker.ev, 0);
        if (r == WAIT_OBJECT_0) {
            recv_packet(&app->to_worker, &pkt, 1);
            if (pkt.tp == PKT_FINI_REQUEST) {
                return -2;
            }
        }
    }

    _splitpath(src_path,
               drive,
               dir,
               fname,
               ext);

    if ((strcmp(ext,".png") != 0) &&
        (strcmp(ext,".jpg") != 0))
    {
        return 0;
    }

    if (strncmp(fname, "mai_", 4) == 0) {
        return 0;
    }

    path_len = strlen(src_path);
    dst_path = malloc(path_len + 5);

    r = _snprintf(dst_path, path_len+5,
                  "%s%smai_%s.png",
                  drive, dir, fname);
    if (r > path_len+5) {
        free(dst_path);
        return 0;
    }

    finder = FindFirstFile(src_path, &orig_st);
    if (finder == INVALID_HANDLE_VALUE) {
        free(dst_path);
        return 0;
    }
    FindClose(finder);

    finder = FindFirstFile(dst_path, &mai_st);
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

    r = w2xconv_convert_file(c, dst_path, src_path, 1, 2.0, 0);
    free(dst_path);

    pkt.tp = PKT_PROGRESS;
    pkt.u.progress.path = strdup(src_path);
    pkt.u.progress.flops = c->flops.flop / c->flops.process_sec;

    send_packet(&app->from_worker, &pkt);

    return r;
}

static int
run_dir(struct app *app, struct W2XConv *c, const char *src_path)
{
    WIN32_FIND_DATA fd;
    HANDLE finder;

    char pattern[MAX_PATH] = "";
    char sub[MAX_PATH] = "";
    int r = 0;

    PathCombine(pattern, src_path, "*");

    finder = FindFirstFile(pattern, &fd);
    if (finder != INVALID_HANDLE_VALUE) {
        do {
            if ((strcmp(fd.cFileName,".") == 0) ||
                (strcmp(fd.cFileName,"..") == 0))
            {
                /* ignore */
            } else {
                PathCombine(sub, src_path, fd.cFileName);

                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    r = run_dir(app, c, sub);
                } else {
                    r = run1(app, c, sub);
                }

                if (r < 0) {
                    break;
                }
            }
        } while(FindNextFile(finder, &fd));

        FindClose(finder);
    }

    return r;
}

static unsigned int __stdcall
proc_thread(void *ap)
{
    struct app *app = (struct app*)ap;
    char *path = app->src_path;

    struct W2XConv *c;
    int r;
    char *src_path, *fullpath, *filepart;
    DWORD attr, pathlen;
    int ret = 1;
    size_t path_len = 4;
    char *self_path = (char*)malloc(path_len+1);
    DWORD len;

    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    c = w2xconv_init(1, 0, 0);
    while (1) {
        len = GetModuleFileName(NULL, self_path, path_len);
        if (len > 0 && len != path_len) {
            break;
        }

        path_len *= 2;
        self_path = (char*)realloc(self_path, path_len+1);
    }

    {
        int cur;
        for (cur=strlen(self_path); cur>=0; cur--) {
            if (self_path[cur] == '\\') {
                self_path[cur] = '\0';
                break;
            }
        }

        char *models_path = malloc(strlen(self_path) + 7 + 1);

        sprintf(models_path, "%s\\models_rgb", self_path);
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
        pkt.u.start.dev_name = strdup(c->target_processor.dev_name);
        send_packet(&app->from_worker, &pkt);
    }

    pathlen = GetFullPathName(path, 0, NULL, NULL);
    if (pathlen == 0) {
        goto fini;
    }

    fullpath = malloc(pathlen);
    GetFullPathName(path, pathlen, fullpath, &filepart);

    attr = GetFileAttributes(fullpath);
    if (attr < 0) {
        MessageBox(NULL, "open failed", path, MB_OK);
        goto fini;
    }

    if (attr & FILE_ATTRIBUTE_DIRECTORY) {
        r = run_dir(app, c, fullpath);
    } else {
        r = run1(app, c, fullpath);
    }

    if (r < 0) {
        goto error;
    }

    ret = 0;
    goto fini;

error:
    if (r != -2) {
        char *err = w2xconv_strerror(&c->last_error);
        MessageBox(NULL, err, "error", MB_OK);
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
        struct packet pkt;
        pkt.tp = PKT_FINI_REQUEST;

        send_packet(&app->to_worker, &pkt);
    }

}

static BOOL
on_create(HWND wnd, LPCREATESTRUCT cp)
{
    struct app *app = cp->lpCreateParams;
    unsigned threadID;
    SetWindowLongPtr(wnd, GWLP_USERDATA, (LONG_PTR)app);

    app->worker_thread = (HANDLE)_beginthreadex(NULL, 0, proc_thread, app, 0, &threadID);

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
    if (app->cur_path == NULL && app->state == STATE_FINI) {
        static const char msg[] = "nop";
        TextOut(dc, 10, 10, msg, sizeof(msg)-1);
    } else {

        char line[4096];
        char path[128];
        size_t len = strlen(app->cur_path), line_len;
        //HGDIOBJ old_brush;
        RECT r;

        if (len > 30) {
            size_t rem = len - 30;
            _snprintf(path, 128, "...%s", app->cur_path + rem);
        } else {
            strcpy(path, app->cur_path);
        }

        if (app->state == STATE_FINI) {
            _snprintf(line, sizeof(line),
                      "%.2f[GFLOPS] %s [Complete!!]",
                      app->cur_flops/(1000.0*1000.0*1000.0),
                      path);
        } else {
            _snprintf(line, sizeof(line),
                      "%.2f[GFLOPS] %s",
                      app->cur_flops/(1000*1000*1000.0),
                      path);
        }

        line_len = strlen(line);

        GetClientRect(app->win, &r);
        FillRect(dc, &r, GetStockBrush(WHITE_BRUSH));

        TextOut(dc, 10, 10, app->dev_name, strlen(app->dev_name));
        TextOut(dc, 10, 28, line, line_len);
    }
    ReleaseDC(app->win, dc);
}

#define WINMAIN

#ifdef WINMAIN
int WINAPI
WinMain(HINSTANCE hInst,
        HINSTANCE hPrev,
        LPSTR lpCmdLine,
        int nCmdShow)
#else
int main(int argc, char **argv)
#endif
{
#ifdef WINMAIN
    int argc = __argc;
    char **argv = __argv;
#else
    HINSTANCE hInst = GetModuleHandle(NULL);
#endif

    WNDCLASSEX cls={0};
    struct app app;
    HWND win;
    int run = 1;

    if (__argc < 2) {
        MessageBox(NULL, "usage : w2xc <in>", "w2xcr", MB_OK);
        return 1;
    }

    app.src_path = __argv[1];

    cls.cbSize = sizeof(cls);
    cls.style = CS_CLASSDC;
    cls.lpfnWndProc = wnd_proc;
    cls.hInstance = hInst;
    cls.hIcon = NULL;
    cls.hCursor = LoadCursor(NULL, (LPCSTR)IDC_ARROW);
    cls.hbrBackground = NULL;
    cls.lpszMenuName = NULL;
    cls.lpszClassName = "w2xcr";
    cls.hIconSm = NULL;

    RegisterClassEx(&cls);

    queue_init(&app.to_worker);
    queue_init(&app.from_worker);

    win = CreateWindowEx(0,
                         "w2xcr",
                         "w2xcr",
                         WS_OVERLAPPEDWINDOW,
                         CW_USEDEFAULT,
                         CW_USEDEFAULT,
                         WIN_WIDTH,
                         WIN_HEIGHT,
                         NULL,
                         NULL,
                         hInst,
                         &app);

    app.win = win;

    ShowWindow(win, SW_SHOW);
    UpdateWindow(win);

    app.state = STATE_RUN;
    app.cur_path = NULL;

    while (run) {
        HANDLE list[1] = {app.from_worker.ev};

        DWORD r = MsgWaitForMultipleObjects(1, list, FALSE, INFINITE, QS_ALLEVENTS);

        if (r == WAIT_OBJECT_0) {
            struct packet pkt;

            while (1) {
                int r = recv_packet(&app.from_worker, &pkt, 0);
                if (r < 0) {
                    break;
                }

                switch (pkt.tp) {
                case PKT_START:
                    app.dev_name = pkt.u.start.dev_name;
                    break;

                case PKT_PROGRESS:
                    app.cur_flops = pkt.u.progress.flops;
                    free(app.cur_path);
                    app.cur_path = pkt.u.progress.path;

                    update_display(&app);

                    break;

                case PKT_FINI_ALL:
                    app.state = STATE_FINI;
                    update_display(&app);
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

    WaitForSingleObject(app.worker_thread, INFINITE);
    CloseHandle(app.worker_thread);

    free(app.dev_name);

    return 0;
}
