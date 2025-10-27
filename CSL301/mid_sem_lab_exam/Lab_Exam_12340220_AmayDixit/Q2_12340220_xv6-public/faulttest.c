#include "types.h"
#include "stat.h"
#include "user.h"

int main(int argc, char *argv[])
{
    printf(1, "Page Faults before: %d\n", getpagefaults());

    int pages = 4;
    int size = pages * 4096;
    char *mem = sbrk(size);
    for (int i = 0; i < size; i += 4096)
        mem[i] = 1;

    printf(1, "Page Faults after: %d\n", getpagefaults());

    exit();
}