#include <windows.h>
#include <tchar.h>

int _tmain(int argc,
           TCHAR **argv)
{
    int i;
    for (i=1; i<argc; i++) {
        if (argv[i][0]=='-') {
            continue;
        }

        //_putts(argv[i]);
        DeleteFile(argv[i]);
    }

    return 0;
}
