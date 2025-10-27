#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
    char *p = malloc(1024*17);
    if (p == NULL){
        perror("malloc");
        return 1;
    }
    printf("Allocated 17KB of memory at address: %p\n", (void*)p);
    printf("Process ID: %d\n", getpid());
    
    for (int i = 0; i < 4; i++) p[10 + i*(4*1024)] = 'a';
    
    printf("Press Enter to exit...\n");
    getchar();
    free(p);
    return 0;
}
