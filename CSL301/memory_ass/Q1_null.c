#include <stdio.h>

int main(void) {
    int *p = NULL;
    printf("About to dereference NULL pointer...\n");
    int val = *p;           // undefined behavior -> usually segfault
    printf("Value: %d\n", val);
    return 0;
}
