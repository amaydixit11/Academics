#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

int main() {
    pid_t pid1 = fork();
    pid_t pid2 = fork();

    printf("PID: %d, Parent PID: %d\n", getpid(), getppid());

    return 0;
}
