#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_INODES 8
#define MAX_BLOCKS 64
#define MAX_FILENAME 32
#define MAX_DESC 64
#define DIRECT_PTRS 4
#define MAX_DIR_ENTRIES 16

struct inode {
    int used;                      // 0 = free, 1 = allocated
    int size;                      // file size in bytes
    int type;                      // 0 = file, 1 = directory
    int creator_id;                // user ID who created file
    time_t created_at;             // creation timestamp
    char description[MAX_DESC];    // short file description
    int direct[DIRECT_PTRS];       // direct data block pointers
};

struct dir_entry {
    char name[MAX_FILENAME];
    int inum;
};
 
struct inode inode_table[MAX_INODES];
int data_block_bitmap[MAX_BLOCKS];   // 0 = free, 1 = used
struct dir_entry directory[MAX_DIR_ENTRIES];
int dir_count = 0;


int alloc_inode() {
    for (int i = 0; i < MAX_INODES; i++) {
        if (!inode_table[i].used) {
            inode_table[i].used = 1;
            inode_table[i].size = 0;
            inode_table[i].type = 0; 
            inode_table[i].creator_id = 0;
            inode_table[i].created_at = time(NULL);
            strcpy(inode_table[i].description, "No description");
            for (int j = 0; j < DIRECT_PTRS; j++)
                inode_table[i].direct[j] = -1;
            return i;
        }
    }
    return -1;  
}

void free_inode(int inum) {
    inode_table[inum].used = 0;
    inode_table[inum].size = 0;
    inode_table[inum].creator_id = 0;
    inode_table[inum].created_at = 0;
    strcpy(inode_table[inum].description, "");
    for (int j = 0; j < DIRECT_PTRS; j++)
        inode_table[inum].direct[j] = -1;
}

int alloc_block() {
    for (int i = 0; i < MAX_BLOCKS; i++) {
        if (data_block_bitmap[i] == 0) {
            data_block_bitmap[i] = 1;
            return i;
        }
    }
    return -1;
}

void free_block(int blocknum) {
    if (blocknum >= 0 && blocknum < MAX_BLOCKS)
        data_block_bitmap[blocknum] = 0;
}

int fs_create(const char *name, int creator_id, const char *desc) {
    for (int i = 0; i < dir_count; i++) {
        if (strcmp(directory[i].name, name) == 0) {
            printf("Error: File '%s' already exists\n", name);
            return -1;
        }
    }

    int inum = alloc_inode();
    if (inum < 0) {
        printf("Error: No free inode available\n");
        return -1;
    }

    // ADD YOUR CODE
   

    
    int block = alloc_block();
    if (block < 0) {
        printf("Error: No free data block available\n");
        free_inode(inum);
        return -1;
    }

    inode_table[inum].direct[0] = block;
    inode_table[inum].size = 0;

    strcpy(directory[dir_count].name, name);
    directory[dir_count].inum = inum;
    dir_count++;

    printf("File '%s' created (inode %d, block %d)\n", name, inum, block);
    return inum;
}

int fs_delete(const char *name) {
    int found = -1;
    for (int i = 0; i < dir_count; i++) {
        if (strcmp(directory[i].name, name) == 0) {
            found = i;
            break;
        }
    }

    if (found < 0) {
        printf("Error: File '%s' not found\n", name);
        return -1;
    }

    int inum = directory[found].inum;

    for (int j = 0; j < DIRECT_PTRS; j++) {
         //ADD YOUR CODE HERE
    }

    free_inode(inum);

    for (int i = found; i < dir_count - 1; i++)
        directory[i] = directory[i + 1];
    dir_count--;

    printf("File '%s' deleted (inode %d freed)\n", name, inum);
    return 0;
}

void fs_show_state() {
    printf("\n--- File System State ---\n");
    printf("Directory entries (%d):\n", dir_count);
    for (int i = 0; i < dir_count; i++)
        printf("  %s → inode %d\n", directory[i].name, directory[i].inum);

    printf("\nInode table:\n");
    for (int i = 0; i < MAX_INODES; i++) {
        if (inode_table[i].used) {
            printf("  inode %d: creator=%d, size=%d, desc='%s'\n",
                   i, inode_table[i].creator_id, inode_table[i].size, inode_table[i].description);
            printf("             created: %s", ctime(&inode_table[i].created_at));
            printf("             blocks: ");
            for (int j = 0; j < DIRECT_PTRS; j++)
                if (inode_table[i].direct[j] != -1)
                    printf("%d ", inode_table[i].direct[j]);
            printf("\n");
        }
    }
    printf("--------------------------\n\n");
}


int main() {
    printf("File Creation and Deletion with Metadata Simulation\n");

    fs_create("file1.txt", 101, "User text document");
    fs_create("report.pdf", 102, "PDF project report");
    fs_create("data.bin", 103, "Binary data log");
    fs_show_state();

    fs_delete("report.pdf");
    fs_show_state();

    fs_create("notes.txt", 104, "Important notes");
    fs_show_state();

    return 0;
}

