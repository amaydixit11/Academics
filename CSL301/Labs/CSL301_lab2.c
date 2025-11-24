#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {
    // Declare and initialize the variable before forking.
    int x = 50;

    printf("Initial value of x: %d (PID: %d)\n", x, (int) getpid());

    // Create a new process by calling fork().
    pid_t rc = fork();

    if (rc < 0) {
        // fork() failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {
        // This is the child process.
        // It gets its own copy of 'x'.
        printf("--- Child process starting (PID: %d) ---\n", (int) getpid());
        x = x + 20;
        printf("Child process: x = %d\n", x);
        printf("--- Child process finishing ---\n");
    } else {
        // This is the parent process.
        // It waits for the child to complete to ensure predictable output order.
        wait(NULL);
        printf("--- Parent process starting (PID: %d) ---\n", (int) getpid());
        // The parent's copy of 'x' is unaffected by the child.
        x = x - 10;
        printf("Parent process: x = %d\n", x);
        printf("--- Parent process finishing ---\n");
    }

    return 0;
}

