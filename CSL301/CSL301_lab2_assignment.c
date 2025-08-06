#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[]) {

    // Create a new process by calling fork().
    pid_t rc = fork();

    if (rc < 0) {
        // fork() failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {
        execl("/bin/ls", "", "-l", (char *) NULL);
    } else {
        printf("Parent finished\n");
    }

    return 0;
}

