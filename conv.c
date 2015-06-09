#include <stdio.h>
#include <string.h>

int
main(int argc, char **argv)
{
    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[2], "w");

    char line[4096];

    while (fgets(line, 4096, in)) {
        int len = strlen(line);

        if (len >= 1 && line[len-1] == '\n') {
            line[len-1] = '\0';
        }

        fprintf(out, "\"%s\\n\"\n", line);
    }
}
