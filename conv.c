#include <stdio.h>
#include <string.h>

int
main(int argc, char **argv)
{
    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[2], "w");

    if (strcmp(argv[3],"str") == 0) {
        char line[4096];
        char buf[8192];

        while (fgets(line, 4096, in)) {
            int len = strlen(line);
            int cur=0, i;

            if (len >= 1 && line[len-1] == '\n') {
                line[len-1] = '\0';
            }

            for (i=0; i<len; i++) {
                char c = line[i];
                if (c == '\\') {
                    buf[cur++] = '\\';
                    buf[cur++] = '\\';
                } else if (c == '"') {
                    buf[cur++] = '\\';
                    buf[cur++] = '"';
                } else {
                    buf[cur++] = line[i];
                }
            }

            buf[cur] = '\0';

            fprintf(out, "\"%s\\n\"\n", buf);
        }
    } else {
        printf("unknown mode : %s\n", argv[3]);
        return 1;
    }

    return 0;
}
