#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/stat.h>


int main(int argc, char *argv[]) {
    int file = open("output.txt", O_CREAT | O_WRONLY | O_APPEND, S_IRWXU);

    // Create a new process by calling fork().
    pid_t rc = fork();

    if (rc < 0) {
        // fork() failed; exit
        fprintf(stderr, "fork failed\n");
        exit(1);
    } else if (rc == 0) {
        write(file, "Child writing\n", 14);
    } else {
        write(file, "Parent writing\n", 15);
    }

    return 0;
}

