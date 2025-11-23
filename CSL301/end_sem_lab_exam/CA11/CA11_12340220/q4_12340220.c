#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_INODES 10
#define MAX_BLOCKS 10
#define DESC_LEN 64

struct inode {
    int in_use;
    int creator_id;
    time_t created_at;
    char description[DESC_LEN];
    int block;
};

struct inode inode_table[MAX_INODES];
int block_used[MAX_BLOCKS] = {0};
char filenames[MAX_INODES][32];

int allocate_inode() {
    for (int i = 0; i < MAX_INODES; i++)
        if (!inode_table[i].in_use) return i;
    return -1;
}

int allocate_block() {
    for (int i = 0; i < MAX_BLOCKS; i++)
        if (!block_used[i]) return i;
    return -1;
}

int find_inode(const char *name) {
    for (int i = 0; i < MAX_INODES; i++)
        if (inode_table[i].in_use && strcmp(filenames[i], name) == 0)
            return i;
    return -1;
}

int create_file(const char *name, int creator_id, const char *desc) {
    int ino = allocate_inode();
    int blk = allocate_block();
    if (ino < 0 || blk < 0) return -1;

    inode_table[ino].in_use = 1;
    inode_table[ino].creator_id = creator_id;
    inode_table[ino].block = blk;
    time(&inode_table[ino].created_at);
    strncpy(inode_table[ino].description, desc, DESC_LEN - 1);

    block_used[blk] = 1;
    strcpy(filenames[ino], name);

    return 0;
}

int delete_file(const char *name) {
    int ino = find_inode(name);
    if (ino < 0) return -1;

    block_used[inode_table[ino].block] = 0;
    inode_table[ino].in_use = 0;
    memset(filenames[ino], 0, 32);

    return 0;
}

void show_inodes() {
    for (int i = 0; i < MAX_INODES; i++) {
        if (inode_table[i].in_use) {
            char tbuf[64];
            struct tm *tm = localtime(&inode_table[i].created_at);
            strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", tm);

            printf("File: %s  | Creator: %d | Created: %s | Desc: %s\n",
                filenames[i],
                inode_table[i].creator_id,
                tbuf,
                inode_table[i].description);
        }
    }
}

int main() {
    create_file("a.txt", 101, "Test file A");
    create_file("b.txt", 102, "Another one");

    printf("After creation:\n");
    show_inodes();

    delete_file("a.txt");

    printf("\nAfter deletion:\n");
    show_inodes();

    return 0;
}
