#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>

int main() {
    int fd = open("output.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    pid_t pid = fork();

    if (pid == 0) {
        write(fd, "12340220 - Child wrote this.\n", 29);
        printf("Child: Wrote to file.\n");
    }
    else {
        write(fd, "Amay Dixit - Parent wrote this.\n", 32);
        printf("Parent: Wrote to file.\n");
    }

    close(fd);
    return 0;
}
