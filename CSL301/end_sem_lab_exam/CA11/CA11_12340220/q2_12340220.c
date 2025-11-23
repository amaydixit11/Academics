    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <dirent.h>
    #include <sys/stat.h>
    #include <unistd.h>
    #include <time.h>
    #include <limits.h>

    void print_permissions(mode_t m) {
        char type = S_ISDIR(m) ? 'd' : '-';
        printf("%c", type);
        printf((m & S_IRUSR) ? "r" : "-");
        printf((m & S_IWUSR) ? "w" : "-");
        printf((m & S_IXUSR) ? "x" : "-");
        printf((m & S_IRGRP) ? "r" : "-");
        printf((m & S_IWGRP) ? "w" : "-");
        printf((m & S_IXGRP) ? "x" : "-");
        printf((m & S_IROTH) ? "r" : "-");
        printf((m & S_IWOTH) ? "w" : "-");
        printf((m & S_IXOTH) ? "x" : "-");
    }

    void print_long(const char *path, const char *name) {
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, name);

        struct stat st;
        if (stat(full, &st) == -1) {
            perror("stat");
            return;
        }

        print_permissions(st.st_mode);
        printf(" %ld %d %d %lld ",
            (long)st.st_nlink,
            st.st_uid, st.st_gid,
            (long long)st.st_size);

        char tbuf[64];
        struct tm *tm = localtime(&st.st_mtime);
        strftime(tbuf, sizeof(tbuf), "%b %d %H:%M", tm);

        printf("%s %s\n", tbuf, name);
    }

    int main(int argc, char *argv[]) {
        int long_flag = 0;
        char *path = ".";

        if (argc >= 2) {
            if (strcmp(argv[1], "-l") == 0) {
                long_flag = 1;
                if (argc >= 3) path = argv[2];
            } else {
                path = argv[1];
            }
        }

        DIR *dir = opendir(path);
        if (!dir) {
            perror("opendir");
            return 1;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (!long_flag)
                printf("%s\n", entry->d_name);
            else
                print_long(path, entry->d_name);
        }

        closedir(dir);
        return 0;
    }
