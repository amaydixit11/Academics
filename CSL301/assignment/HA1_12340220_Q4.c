#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main() {
    pid_t pid1, pid2;

    pid1 = fork();
    if (pid1 == 0) {
        printf("First Child: PID = %d, PPID = %d, I am first child\n", getpid(), getppid());
        return 0;
    }

    pid2 = fork();
    if (pid2 == 0) {
        printf("Second Child: PID = %d, PPID = %d, I am second child\n", getpid(), getppid());
        return 0;
    }

    printf("Parent: PID = %d, I am Amay Dixit\n", getpid());
    return 0;
}
