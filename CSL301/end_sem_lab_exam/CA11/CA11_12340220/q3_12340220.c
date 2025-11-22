    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <errno.h>

    int copy_file(const char *src, const char *dst) {
        int in_fd, out_fd;
        struct stat st;

        if (stat(src, &st) == -1) {
            perror("stat");
            return -1;
        }

        in_fd = open(src, O_RDONLY);
        if (in_fd < 0) {
            perror("open src");
            return -1;
        }

        out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, st.st_mode & 0777);
        if (out_fd < 0) {
            perror("open dst");
            close(in_fd);
            return -1;
        }

        char buf[4096];
        ssize_t n;
        while ((n = read(in_fd, buf, sizeof(buf))) > 0) {
            if (write(out_fd, buf, n) != n) {
                perror("write");
                close(in_fd);
                close(out_fd);
                return -1;
            }
        }

        close(in_fd);
        close(out_fd);
        return (n < 0) ? -1 : 0;
    }

    int do_mv(const char *src, const char *dst) {
        if (rename(src, dst) == 0) return 0;

        if (errno == EXDEV) {
            if (copy_file(src, dst) == 0) {
                if (unlink(src) == -1) perror("unlink");
                return 0;
            }
        }
        perror("mv");
        return -1;
    }

    char *basename(char *path) {
        char *b = strrchr(path, '/');
        return b ? b + 1 : path;
    }

    int main(int argc, char *argv[]) {
        if (argc < 3) {
            fprintf(stderr,"Usage: ./cp file1 file2  OR  ./mv file1 file2  OR  ./a.out cp file1 file2\n");
            return 1;
        }

        char *cmd;
        int offset = 0;

        char *prog = basename(argv[0]);
        if (strcmp(prog, "cp") == 0 || strcmp(prog, "mv") == 0) {
            cmd = prog;
        } else {
            cmd = argv[1];
            offset = 1;
        }

        if (argc - offset != 3) {
            fprintf(stderr,"Invalid args.\n");
            return 1;
        }

        char *src = argv[1 + offset];
        char *dst = argv[2 + offset];

        if (strcmp(cmd, "cp") == 0) {
            return copy_file(src,dst);
        } else if (strcmp(cmd, "mv") == 0) {
            return do_mv(src,dst);
        } else {
            fprintf(stderr,"Unknown command.\n");
            return 1;
        }
    }
