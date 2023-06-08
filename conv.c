#include <stdio.h>
#include <string.h>

int
main(int argc, char **argv)
{
    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[2], "w");

    if (strcmp(argv[3],"str") == 0) {
        fputc('{', out);
        fputc('\n', out);
        char line[4096];

        while (fgets(line, 4096, in)) {
            size_t len = strlen(line);
            size_t i;

            for (i=0; i<len; i++) {
                fprintf(out, "0x%02x,", line[i]);
            }
            fputc('\n', out);
        }

        fputs("0x00, }", out);
        fputc('\n', out);
    } else {
        printf("unknown mode : %s\n", argv[3]);
        return 1;
    }

    return 0;
}
