#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>

void explore(const char *path, int depth, int level){
    if (level > depth) return;
    DIR *d = opendir(path);

    if (!d){
        for (int i = 0; i < level * 4; i++) printf("  ");
        printf("%s [ACCESS DENIED]\n", path);
        return;
    }
    struct dirent *de;
    char newpath[512];
    
    while ((de = readdir(d)) != NULL) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        for (int i = 0; i < level * 4; i++) printf("  ");
        snprintf(newpath, sizeof(newpath), "%s/%s", path, de->d_name);
        struct stat st;
        lstat(newpath, &st);
        printf("%s", de->d_name);
        if (S_ISREG(st.st_mode)) printf(" [FILE] (size: %lld bytes)\n", (long long)st.st_size);
        else if (S_ISDIR(st.st_mode)) printf(" [DIR] (size: %lld bytes)\n", (long long)st.st_size);
        else if (S_ISLNK(st.st_mode)) printf(" [LINK] (size: %lld bytes)\n", (long long)st.st_size);
        else printf(" [OTHER] (size: %lld bytes)\n", (long long)st.st_size);
        if (S_ISDIR(st.st_mode)) {
            explore(newpath, depth, level + 1);
        }
    }

    closedir(d);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <directory> <depth>\n", argv[0]);
        return 1;
    }
    int depth = atoi(argv[2]);
    printf("%s\n", argv[1]);
    if (depth >= 0) explore(argv[1], depth, 1);
    return 0;
}