#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int global_var = 10;

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork failed");
        return 1;
    }
    else if (pid == 0) {
        global_var += 10;
        printf("Child Process: global_var = %d\n", global_var);
        printf("Child: My changes do not affect the parent.\n");
    }
    else {
        global_var += 5;
        printf("Parent Process: global_var = %d\n", global_var);
        printf("Parent: My changes do not affect the child.\n");
    }
    return 0;
}
