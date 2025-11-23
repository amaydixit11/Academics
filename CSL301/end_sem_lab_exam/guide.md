# Ultimate OS Lab Exam Guide - CSL301
## Operating Systems Lab - Complete Reference

---

## Table of Contents
1. [POSIX Threads (Pthreads)](#posix-threads)
2. [Synchronization Primitives](#synchronization-primitives)
3. [Classical Synchronization Problems](#classical-problems)
4. [Condition Variables](#condition-variables)
5. [File System Operations](#file-system-operations)
6. [Process Control](#process-control)
7. [Common Code Patterns](#common-patterns)
8. [Debugging Tips](#debugging-tips)

---

## 1. POSIX Threads (Pthreads) {#posix-threads}

### Essential Headers
```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
```

### Thread Types and Functions

#### pthread_t - Thread Handle
```c
pthread_t thread_id;
pthread_t threads[10];  // Array of threads
```

#### pthread_create() - Create a Thread
```c
int pthread_create(
    pthread_t *thread,           // Pointer to thread ID
    const pthread_attr_t *attr,  // Thread attributes (usually NULL)
    void *(*start_routine)(void*), // Thread function
    void *arg                    // Argument to thread function
);

// Example:
pthread_t tid;
int thread_num = 5;
pthread_create(&tid, NULL, thread_function, &thread_num);
```

#### pthread_join() - Wait for Thread Completion
```c
int pthread_join(
    pthread_t thread,    // Thread to wait for
    void **retval       // Pointer to store return value
);

// Example 1: No return value
pthread_join(tid, NULL);

// Example 2: Collecting return value
void *result;
pthread_join(tid, &result);
int value = *(int*)result;
free(result);
```

#### pthread_exit() - Exit Thread
```c
void pthread_exit(void *retval);

// Example:
void* thread_func(void* arg) {
    int* result = malloc(sizeof(int));
    *result = 42;
    pthread_exit(result);  // or: return result;
}
```

#### pthread_cancel() - Cancel Thread
```c
pthread_cancel(thread_id);
```

### Thread Function Patterns

#### Basic Thread Function
```c
void* thread_function(void* arg) {
    // Cast argument
    int id = *(int*)arg;
    
    // Do work
    printf("Thread %d running\n", id);
    
    return NULL;
}
```

#### Thread with Return Value
```c
void* compute_square(void* arg) {
    int n = *(int*)arg;
    int* result = malloc(sizeof(int));
    *result = n * n;
    return result;
}

// In main:
void* result;
pthread_join(tid, &result);
printf("Result: %d\n", *(int*)result);
free(result);
```

#### Thread with Struct Argument
```c
typedef struct {
    int start;
    int end;
    int* array;
    int result;
} ThreadArgs;

void* process_segment(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int sum = 0;
    for(int i = args->start; i < args->end; i++) {
        sum += args->array[i];
    }
    args->result = sum;
    return NULL;
}
```

### Common Thread Patterns

#### Creating Multiple Threads
```c
#define NUM_THREADS 10

pthread_t threads[NUM_THREADS];
int thread_ids[NUM_THREADS];

for(int i = 0; i < NUM_THREADS; i++) {
    thread_ids[i] = i;
    pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]);
}

for(int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
}
```

#### Master-Worker Pattern
```c
// Worker threads
for(int i = 0; i < N_WORKERS; i++) {
    pthread_create(&workers[i], NULL, worker_func, &ids[i]);
}

// Master thread
pthread_create(&master, NULL, master_func, NULL);

// Wait for master
pthread_join(master, NULL);

// Wait for workers
for(int i = 0; i < N_WORKERS; i++) {
    pthread_join(workers[i], NULL);
}
```

---

## 2. Synchronization Primitives {#synchronization-primitives}

### Mutex (Mutual Exclusion)

#### Declaration and Initialization
```c
pthread_mutex_t lock;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;  // Static init

// Dynamic initialization
pthread_mutex_init(&lock, NULL);

// Destroy when done
pthread_mutex_destroy(&lock);
```

#### Lock and Unlock
```c
pthread_mutex_lock(&lock);
// Critical section
pthread_mutex_unlock(&lock);
```

#### Complete Example
```c
#include <pthread.h>
#include <stdio.h>

int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void* increment(void* arg) {
    for(int i = 0; i < 100000; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("Final counter: %d\n", counter);
    return 0;
}
```

### Semaphores

#### Declaration and Initialization
```c
#include <semaphore.h>

sem_t sem;

// Initialize semaphore
sem_init(
    &sem,        // Semaphore pointer
    0,           // 0 = shared between threads (not processes)
    initial_val  // Initial value
);

// Destroy when done
sem_destroy(&sem);
```

#### Operations
```c
// Wait (decrement, P operation)
sem_wait(&sem);  // Blocks if value is 0

// Signal (increment, V operation)
sem_post(&sem);  // Increments value, wakes waiting thread

// Try wait (non-blocking)
int sem_trywait(&sem);  // Returns -1 if would block
```

#### Semaphore Patterns

**Binary Semaphore (Signaling)**
```c
sem_t signal;
sem_init(&signal, 0, 0);  // Start at 0

// Thread A
printf("A done\n");
sem_post(&signal);  // Signal

// Thread B
sem_wait(&signal);  // Wait for signal
printf("B starts\n");
```

**Counting Semaphore (Resource Pool)**
```c
#define MAX_RESOURCES 3
sem_t resources;
sem_init(&resources, 0, MAX_RESOURCES);

void* worker(void* arg) {
    sem_wait(&resources);     // Acquire resource
    // Use resource
    printf("Using resource\n");
    sleep(1);
    sem_post(&resources);     // Release resource
    return NULL;
}
```

**Multiplex Pattern (Limited Concurrent Access)**
```c
#define MAX_CONCURRENT 3
sem_t multiplex;
sem_init(&multiplex, 0, MAX_CONCURRENT);

void* thread_func(void* arg) {
    sem_wait(&multiplex);
    printf("Entered critical section\n");
    // Critical section
    sem_post(&multiplex);
    return NULL;
}
```

---

## 3. Classical Synchronization Problems {#classical-problems}

### Producer-Consumer Problem

#### Unbounded Buffer (Basic)
```c
#define BUFFER_SIZE 100

sem_t items;
pthread_mutex_t mutex;

int buffer[BUFFER_SIZE];
int fill_ptr = 0;
int use_ptr = 0;

void init() {
    sem_init(&items, 0, 0);
    pthread_mutex_init(&mutex, NULL);
}

void* producer(void* arg) {
    while(1) {
        int item = produce_item();
        
        pthread_mutex_lock(&mutex);
        buffer[fill_ptr] = item;
        fill_ptr = (fill_ptr + 1) % BUFFER_SIZE;
        pthread_mutex_unlock(&mutex);
        
        sem_post(&items);
    }
}

void* consumer(void* arg) {
    while(1) {
        sem_wait(&items);
        
        pthread_mutex_lock(&mutex);
        int item = buffer[use_ptr];
        use_ptr = (use_ptr + 1) % BUFFER_SIZE;
        pthread_mutex_unlock(&mutex);
        
        consume_item(item);
    }
}
```

#### Bounded Buffer (Complete Solution)
```c
#define BUFFER_SIZE 10

sem_t empty, full;
pthread_mutex_t mutex;
int buffer[BUFFER_SIZE];
int fill_ptr = 0, use_ptr = 0;

void init() {
    sem_init(&empty, 0, BUFFER_SIZE);  // All slots empty
    sem_init(&full, 0, 0);             // No items
    pthread_mutex_init(&mutex, NULL);
}

void* producer(void* arg) {
    while(1) {
        int item = produce_item();
        
        sem_wait(&empty);           // Wait for empty slot
        pthread_mutex_lock(&mutex);
        buffer[fill_ptr] = item;
        fill_ptr = (fill_ptr + 1) % BUFFER_SIZE;
        pthread_mutex_unlock(&mutex);
        sem_post(&full);            // Signal item available
    }
}

void* consumer(void* arg) {
    while(1) {
        sem_wait(&full);            // Wait for item
        pthread_mutex_lock(&mutex);
        int item = buffer[use_ptr];
        use_ptr = (use_ptr + 1) % BUFFER_SIZE;
        pthread_mutex_unlock(&mutex);
        sem_post(&empty);           // Signal slot available
        
        consume_item(item);
    }
}
```

### Readers-Writers Problem

#### Basic Solution (Readers Preference)
```c
sem_t roomEmpty;
pthread_mutex_t readMutex;
int readers = 0;

void init() {
    sem_init(&roomEmpty, 0, 1);
    pthread_mutex_init(&readMutex, NULL);
}

void* reader(void* arg) {
    while(1) {
        pthread_mutex_lock(&readMutex);
        readers++;
        if(readers == 1)
            sem_wait(&roomEmpty);  // First reader locks writers
        pthread_mutex_unlock(&readMutex);
        
        // READ DATA
        
        pthread_mutex_lock(&readMutex);
        readers--;
        if(readers == 0)
            sem_post(&roomEmpty);  // Last reader unlocks writers
        pthread_mutex_unlock(&readMutex);
    }
}

void* writer(void* arg) {
    while(1) {
        sem_wait(&roomEmpty);
        // WRITE DATA
        sem_post(&roomEmpty);
    }
}
```

#### Lightswitch Pattern
```c
typedef struct {
    int counter;
    sem_t mutex;
} Lightswitch;

void lightswitch_init(Lightswitch* ls) {
    ls->counter = 0;
    sem_init(&ls->mutex, 0, 1);
}

void lightswitch_lock(Lightswitch* ls, sem_t* semaphore) {
    sem_wait(&ls->mutex);
    ls->counter++;
    if(ls->counter == 1)
        sem_wait(semaphore);
    sem_post(&ls->mutex);
}

void lightswitch_unlock(Lightswitch* ls, sem_t* semaphore) {
    sem_wait(&ls->mutex);
    ls->counter--;
    if(ls->counter == 0)
        sem_post(semaphore);
    sem_post(&ls->mutex);
}

// Usage:
Lightswitch readSwitch;
sem_t roomEmpty;

void* reader(void* arg) {
    lightswitch_lock(&readSwitch, &roomEmpty);
    // READ
    lightswitch_unlock(&readSwitch, &roomEmpty);
}

void* writer(void* arg) {
    sem_wait(&roomEmpty);
    // WRITE
    sem_post(&roomEmpty);
}
```

#### Writers Priority Solution
```c
pthread_mutex_t lock;
pthread_cond_t can_read, can_write;
int read_count = 0;
int write_count = 0;
int waiting_writers = 0;

void start_read() {
    pthread_mutex_lock(&lock);
    while(write_count == 1 || waiting_writers > 0) {
        pthread_cond_wait(&can_read, &lock);
    }
    read_count++;
    pthread_mutex_unlock(&lock);
}

void end_read() {
    pthread_mutex_lock(&lock);
    read_count--;
    if(read_count == 0)
        pthread_cond_signal(&can_write);
    pthread_mutex_unlock(&lock);
}

void start_write() {
    pthread_mutex_lock(&lock);
    waiting_writers++;
    while(read_count > 0 || write_count == 1) {
        pthread_cond_wait(&can_write, &lock);
    }
    waiting_writers--;
    write_count = 1;
    pthread_mutex_unlock(&lock);
}

void end_write() {
    pthread_mutex_lock(&lock);
    write_count = 0;
    if(waiting_writers > 0)
        pthread_cond_signal(&can_write);
    else
        pthread_cond_broadcast(&can_read);
    pthread_mutex_unlock(&lock);
}
```

### Dining Philosophers Problem

#### Deadlock Version (DO NOT USE)
```c
#define N 5
sem_t forks[N];

void init() {
    for(int i = 0; i < N; i++)
        sem_init(&forks[i], 0, 1);
}

void* philosopher(void* arg) {
    int id = *(int*)arg;
    int left = id;
    int right = (id + 1) % N;
    
    while(1) {
        // Think
        
        sem_wait(&forks[left]);   // DEADLOCK RISK!
        sem_wait(&forks[right]);
        
        // Eat
        
        sem_post(&forks[left]);
        sem_post(&forks[right]);
    }
}
```

#### Solution: Lower ID First
```c
void* philosopher(void* arg) {
    int id = *(int*)arg;
    int left = id;
    int right = (id + 1) % N;
    
    while(1) {
        // Think
        
        // Pick up lower numbered fork first
        if(left < right) {
            sem_wait(&forks[left]);
            sem_wait(&forks[right]);
        } else {
            sem_wait(&forks[right]);
            sem_wait(&forks[left]);
        }
        
        // Eat
        
        sem_post(&forks[left]);
        sem_post(&forks[right]);
    }
}
```

### Barrier Synchronization

#### Non-Reusable Barrier (Simple)
```c
int count = 0;
pthread_mutex_t mutex;
sem_t barrier;

void init(int n) {
    pthread_mutex_init(&mutex, NULL);
    sem_init(&barrier, 0, 0);
}

void* thread_func(void* arg) {
    // Rendezvous
    pthread_mutex_lock(&mutex);
    count++;
    if(count == N)
        sem_post(&barrier);
    pthread_mutex_unlock(&mutex);
    
    sem_wait(&barrier);
    sem_post(&barrier);  // Let others through
    
    // Critical point
}
```

#### Reusable Barrier (Two Turnstiles)
```c
int count = 0;
pthread_mutex_t mutex;
sem_t turnstile1, turnstile2;

void init() {
    pthread_mutex_init(&mutex, NULL);
    sem_init(&turnstile1, 0, 0);
    sem_init(&turnstile2, 0, 1);
}

void* thread_func(void* arg) {
    // Phase 1: Arrival
    pthread_mutex_lock(&mutex);
    count++;
    if(count == N) {
        sem_wait(&turnstile2);
        sem_post(&turnstile1);
    }
    pthread_mutex_unlock(&mutex);
    
    sem_wait(&turnstile1);
    sem_post(&turnstile1);
    
    // Critical point
    
    // Phase 2: Departure
    pthread_mutex_lock(&mutex);
    count--;
    if(count == 0) {
        sem_wait(&turnstile1);
        sem_post(&turnstile2);
    }
    pthread_mutex_unlock(&mutex);
    
    sem_wait(&turnstile2);
    sem_post(&turnstile2);
}
```

---

## 4. Condition Variables {#condition-variables}

### Basic Condition Variable Operations

#### Declaration and Initialization
```c
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

// Or dynamic:
pthread_mutex_init(&lock, NULL);
pthread_cond_init(&cond, NULL);

// Destroy when done:
pthread_cond_destroy(&cond);
pthread_mutex_destroy(&lock);
```

#### Wait Pattern (ALWAYS use while loop)
```c
pthread_mutex_lock(&lock);

while(condition_not_met) {
    pthread_cond_wait(&cond, &lock);
    // Atomically: unlocks mutex, waits, then re-locks mutex
}

// Condition is now met, do work
// ...

pthread_mutex_unlock(&lock);
```

#### Signal and Broadcast
```c
pthread_mutex_lock(&lock);

// Change state
state = new_value;

pthread_cond_signal(&cond);     // Wake one waiting thread
// OR
pthread_cond_broadcast(&cond);  // Wake all waiting threads

pthread_mutex_unlock(&lock);
```

### Common Condition Variable Patterns

#### Ordered Thread Execution (A → B → C)
```c
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int turn = 0;

void* printA(void* arg) {
    for(int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);
        while(turn != 0)
            pthread_cond_wait(&cond, &lock);
        
        printf("A ");
        turn = 1;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&lock);
    }
}

void* printB(void* arg) {
    for(int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);
        while(turn != 1)
            pthread_cond_wait(&cond, &lock);
        
        printf("B ");
        turn = 2;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&lock);
    }
}

void* printC(void* arg) {
    for(int i = 0; i < N; i++) {
        pthread_mutex_lock(&lock);
        while(turn != 2)
            pthread_cond_wait(&cond, &lock);
        
        printf("C\n");
        turn = 0;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&lock);
    }
}
```

#### Producer-Consumer with Condition Variables
```c
#define MAX_SIZE 10

int buffer[MAX_SIZE];
int count = 0;
pthread_mutex_t lock;
pthread_cond_t not_full, not_empty;

void* producer(void* arg) {
    while(1) {
        int item = produce_item();
        
        pthread_mutex_lock(&lock);
        while(count == MAX_SIZE)
            pthread_cond_wait(&not_full, &lock);
        
        buffer[count++] = item;
        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&lock);
    }
}

void* consumer(void* arg) {
    while(1) {
        pthread_mutex_lock(&lock);
        while(count == 0)
            pthread_cond_wait(&not_empty, &lock);
        
        int item = buffer[--count];
        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&lock);
        
        consume_item(item);
    }
}
```

#### Thread Sleep/Wakeup Mechanism
```c
struct thread_event {
    pthread_mutex_t m;
    pthread_cond_t c;
    int awake;
};

struct thread_event events[N];

void* sleeper(void* arg) {
    int id = *(int*)arg;
    
    pthread_mutex_lock(&events[id].m);
    while(!events[id].awake) {
        pthread_cond_wait(&events[id].c, &events[id].m);
    }
    printf("Thread %d woke up\n", id);
    pthread_mutex_unlock(&events[id].m);
}

void* waker(void* arg) {
    sleep(2);
    
    for(int i = 0; i < N; i++) {
        pthread_mutex_lock(&events[i].m);
        events[i].awake = 1;
        pthread_cond_signal(&events[i].c);
        pthread_mutex_unlock(&events[i].m);
    }
}
```

#### Job Dispatcher Pattern
```c
#define MAX_JOBS 5

int jobs[MAX_JOBS];
int count = 0;
pthread_mutex_t lock;
pthread_cond_t not_full, not_empty;

void* dispatcher(void* arg) {
    for(int job_id = 1; job_id <= 10; job_id++) {
        pthread_mutex_lock(&lock);
        while(count == MAX_JOBS)
            pthread_cond_wait(&not_full, &lock);
        
        jobs[count++] = job_id;
        printf("Added job %d\n", job_id);
        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&lock);
    }
}

void* worker(void* arg) {
    int id = *(int*)arg;
    
    while(1) {
        pthread_mutex_lock(&lock);
        while(count == 0)
            pthread_cond_wait(&not_empty, &lock);
        
        int job = jobs[--count];
        printf("Worker %d processing %d\n", id, job);
        pthread_cond_signal(&not_full);
        pthread_mutex_unlock(&lock);
        
        // Process job
    }
}
```

---

## 5. File System Operations {#file-system-operations}

### Directory Operations

#### Headers
```c
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
```

#### Reading Directory Contents
```c
DIR* opendir(const char* name);
struct dirent* readdir(DIR* dirp);
int closedir(DIR* dirp);

// Example: List files
DIR* dir = opendir(".");
if(!dir) {
    perror("opendir");
    return 1;
}

struct dirent* entry;
while((entry = readdir(dir)) != NULL) {
    printf("%s\n", entry->d_name);
}

closedir(dir);
```

#### struct dirent
```c
struct dirent {
    ino_t d_ino;           // Inode number
    char d_name[256];      // Filename
    unsigned char d_type;  // File type
};

// d_type values:
// DT_REG  - Regular file
// DT_DIR  - Directory
// DT_LNK  - Symbolic link
```

### File Metadata (stat)

#### struct stat
```c
struct stat {
    mode_t st_mode;      // File mode (permissions + type)
    nlink_t st_nlink;    // Number of hard links
    uid_t st_uid;        // User ID
    gid_t st_gid;        // Group ID
    off_t st_size;       // File size in bytes
    time_t st_mtime;     // Last modification time
    time_t st_atime;     // Last access time
    time_t st_ctime;     // Last status change time
};
```

#### Getting File Information
```c
int stat(const char* path, struct stat* buf);
int lstat(const char* path, struct stat* buf);  // For symlinks

// Example:
struct stat st;
if(stat("file.txt", &st) == -1) {
    perror("stat");
    return 1;
}

printf("Size: %ld bytes\n", (long)st.st_size);
printf("UID: %d\n", st.st_uid);
printf("GID: %d\n", st.st_gid);
```

#### File Type Macros
```c
S_ISREG(st.st_mode)   // Is regular file?
S_ISDIR(st.st_mode)   // Is directory?
S_ISLNK(st.st_mode)   // Is symbolic link?
S_ISFIFO(st.st_mode)  // Is FIFO/pipe?
S_ISSOCK(st.st_mode)  // Is socket?
```

#### Permission Bits
```c
// Owner permissions
st.st_mode & S_IRUSR  // Read
st.st_mode & S_IWUSR  // Write
st.st_mode & S_IXUSR  // Execute

// Group permissions
st.st_mode & S_IRGRP
st.st_mode & S_IWGRP
st.st_mode & S_IXGRP

// Others permissions
st.st_mode & S_IROTH
st.st_mode & S_IWOTH
st.st_mode & S_IXOTH
```

#### Printing Permissions (ls -l style)
```c
void print_permissions(mode_t mode) {
    printf("%c", S_ISDIR(mode) ? 'd' : '-');
    printf("%c", (mode & S_IRUSR) ? 'r' : '-');
    printf("%c", (mode & S_IWUSR) ? 'w' : '-');
    printf("%c", (mode & S_IXUSR) ? 'x' : '-');
    printf("%c", (mode & S_IRGRP) ? 'r' : '-');
    printf("%c", (mode & S_IWGRP) ? 'w' : '-');
    printf("%c", (mode & S_IXGRP) ? 'x' : '-');
    printf("%c", (mode & S_IROTH) ? 'r' : '-');
    printf("%c", (mode & S_IWOTH) ? 'w' : '-');
    printf("%c", (mode & S_IXOTH) ? 'x' : '-');
}
```

#### Formatting Time
```c
char time_buf[64];
struct tm* tm = localtime(&st.st_mtime);
strftime(time_buf, sizeof(time_buf), "%b %d %H:%M", tm);
printf("%s\n", time_buf);
```

### Low-Level File Operations

#### open()
```c
int open(const char* pathname, int flags, mode_t mode);

// Flags:
O_RDONLY   // Read only
O_WRONLY   // Write only
O_RDWR     // Read and write
O_CREAT    // Create if doesn't exist
O_TRUNC    // Truncate to 0 length
O_APPEND   // Append mode

// Example:
int fd = open("file.txt", O_RDONLY);
if(fd < 0) {
    perror("open");
    return 1;
}

// Create new file with permissions
int fd = open("new.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
```

#### read() and write()
```c
ssize_t read(int fd, void* buf, size_t count);
ssize_t write(int fd, const void* buf, size_t count);

// Example: Copy file
char buffer[4096];
ssize_t n;
while((n = read(fd_in, buffer, sizeof(buffer))) > 0) {
    if(write(fd_out, buffer, n) != n) {
        perror("write");
        break;
    }
}
```

#### close()
```c
int close(int fd);

// Always close file descriptors
close(fd);
```

#### File Copy Function
```c
int copy_file(const char* src, const char* dst) {
    int in_fd = open(src, O_RDONLY);
    if(in_fd < 0) return -1;
    
    struct stat st;
    stat(src, &st);
    
    int out_fd = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 
                      st.st_mode & 0777);
    if(out_fd < 0) {
        close(in_fd);
        return -1;
    }
    
    char buf[4096];
    ssize_t n;
    while((n = read(in_fd, buf, sizeof(buf))) > 0) {
        if(write(out_fd, buf, n) != n) {
            close(in_fd);
            close(out_fd);
            return -1;
        }
    }
    
    close(in_fd);
    close(out_fd);
    return 0;
}
```

### File System Operations

#### rename() - Move/Rename File
```c
int rename(const char* old, const char* new);

// Example:
if(rename("old.txt", "new.txt") == -1) {
    perror("rename");
}
```

#### unlink() - Delete File
```c
int unlink(const char* pathname);

// Example:
if(unlink("file.txt") == -1) {
    perror("unlink");
}
```

#### Move File (Cross-filesystem)
```c
int move_file(const char* src, const char* dst) {
    if(rename(src, dst) == 0)
        return 0;
    
    // rename failed, try copy + delete
    if(errno == EXDEV) {  // Cross-device link
        if(copy_file(src, dst) == 0) {
            unlink(src);
            return 0;
        }
    }
    return -1;
}
```

### Simple Inode/File System Simulation

```c
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
int block_used[MAX_BLOCKS];
char filenames[MAX_INODES][32];

int allocate_inode() {
    for(int i = 0; i < MAX_INODES; i++)
        if(!inode_table[i].in_use)
            return i;
    return -1;
}

int allocate_block() {
    for(int i = 0; i < MAX_BLOCKS; i++)
        if(!block_used[i])
            return i;
    return -1;
}

int create_file(const char* name, int creator_id, const char* desc) {
    int ino = allocate_inode();
    int blk = allocate_block();
    
    if(ino < 0 || blk < 0)
        return -1;
    
    inode_table[ino].in_use = 1;
    inode_table[ino].creator_id = creator_id;
    inode_table[ino].block = blk;
    time(&inode_table[ino].created_at);
    strncpy(inode_table[ino].description, desc, DESC_LEN-1);
    
    block_used[blk] = 1;
    strcpy(filenames[ino], name);
    
    return ino;
}

int delete_file(const char* name) {
    int ino = find_inode(name);
    if(ino < 0) return -1;
    
    block_used[inode_table[ino].block] = 0;
    inode_table[ino].in_use = 0;
    memset(filenames[ino], 0, 32);
    
    return 0;
}
```

---

## 6. Process Control {#process-control}

### Basic Process Operations

#### fork() - Create New Process
```c
#include <unistd.h>
#include <sys/types.h>

pid_t fork(void);

// Returns:
// - 0 in child process
// - child PID in parent process
// - -1 on error

// Example:
pid_t pid = fork();

if(pid < 0) {
    perror("fork failed");
} else if(pid == 0) {
    // Child process
    printf("I am child, PID=%d\n", getpid());
} else {
    // Parent process
    printf("I am parent, child PID=%d\n", pid);
}
```

#### wait() and waitpid()
```c
#include <sys/wait.h>

pid_t wait(int* status);
pid_t waitpid(pid_t pid, int* status, int options);

// Example: Wait for any child
int status;
pid_t child_pid = wait(&status);

if(WIFEXITED(status)) {
    printf("Child exited with status %d\n", WEXITSTATUS(status));
}

// Example: Wait for specific child
waitpid(child_pid, &status, 0);

// Non-blocking wait
waitpid(-1, &status, WNOHANG);
```

#### exec() Family
```c
int execl(const char* path, const char* arg, ...);
int execlp(const char* file, const char* arg, ...);
int execv(const char* path, char* const argv[]);
int execvp(const char* file, char* const argv[]);

// Example: Execute command
execl("/bin/ls", "ls", "-l", NULL);

// Example: Execute with array
char* args[] = {"ls", "-l", NULL};
execv("/bin/ls", args);

// Note: exec replaces current process image
// If exec succeeds, it never returns
```

#### exit() and _exit()
```c
void exit(int status);   // Cleanup, flush buffers
void _exit(int status);  // Immediate exit

// Example:
exit(0);  // Success
exit(1);  // Error
```

### Signal Handling

#### signal()
```c
#include <signal.h>

void (*signal(int signum, void (*handler)(int)))(int);

// Example: Catch Ctrl+C
void sigint_handler(int sig) {
    printf("\nCaught signal %d\n", sig);
    exit(0);
}

int main() {
    signal(SIGINT, sigint_handler);
    
    while(1) {
        printf("Running...\n");
        sleep(1);
    }
}
```

#### Common Signals
```c
SIGINT   // Interrupt (Ctrl+C)
SIGTERM  // Termination request
SIGKILL  // Kill (cannot be caught)
SIGSTOP  // Stop (cannot be caught)
SIGCHLD  // Child process terminated
SIGALRM  // Alarm clock
SIGUSR1  // User-defined signal 1
SIGUSR2  // User-defined signal 2
```

#### sigaction() (More Control)
```c
struct sigaction {
    void (*sa_handler)(int);
    sigset_t sa_mask;
    int sa_flags;
};

int sigaction(int signum, const struct sigaction* act,
              struct sigaction* oldact);

// Example:
struct sigaction sa;
sa.sa_handler = handler_func;
sigemptyset(&sa.sa_mask);
sa.sa_flags = 0;

sigaction(SIGINT, &sa, NULL);
```

#### kill() - Send Signal
```c
int kill(pid_t pid, int sig);

// Example: Send SIGTERM to process
kill(child_pid, SIGTERM);

// Send to process group
kill(-pgid, SIGINT);

// Send to self
kill(getpid(), SIGUSR1);
```

### Pipes

#### Creating and Using Pipes
```c
int pipe(int pipefd[2]);
// pipefd[0] - read end
// pipefd[1] - write end

// Example: Parent-child communication
int pipefd[2];
pipe(pipefd);

pid_t pid = fork();

if(pid == 0) {
    // Child: read from pipe
    close(pipefd[1]);  // Close write end
    
    char buffer[100];
    read(pipefd[0], buffer, sizeof(buffer));
    printf("Child received: %s\n", buffer);
    
    close(pipefd[0]);
} else {
    // Parent: write to pipe
    close(pipefd[0]);  // Close read end
    
    char* msg = "Hello from parent";
    write(pipefd[1], msg, strlen(msg) + 1);
    
    close(pipefd[1]);
    wait(NULL);
}
```

---

## 7. Common Code Patterns {#common-patterns}

### Array Segment Processing

#### Parallel Sum with Threads
```c
#define SIZE 100
#define NUM_THREADS 10
#define SEGMENT_SIZE (SIZE / NUM_THREADS)

typedef struct {
    int start;
    int end;
    int* array;
} Args;

int partial_sums[NUM_THREADS];

void* compute_sum(void* arg) {
    Args* a = (Args*)arg;
    int sum = 0;
    
    for(int i = a->start; i < a->end; i++)
        sum += a->array[i];
    
    partial_sums[a->index] = sum;
    free(a);
    return NULL;
}

int main() {
    int array[SIZE];
    pthread_t threads[NUM_THREADS];
    
    // Initialize array
    for(int i = 0; i < SIZE; i++)
        array[i] = i + 1;
    
    // Create worker threads
    for(int i = 0; i < NUM_THREADS; i++) {
        Args* a = malloc(sizeof(Args));
        a->start = i * SEGMENT_SIZE;
        a->end = (i + 1) * SEGMENT_SIZE;
        a->array = array;
        a->index = i;
        pthread_create(&threads[i], NULL, compute_sum, a);
    }
    
    // Wait for all threads
    for(int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);
    
    // Aggregate results
    int total = 0;
    for(int i = 0; i < NUM_THREADS; i++)
        total += partial_sums[i];
    
    printf("Total sum: %d\n", total);
}
```

### Command Line Argument Processing

```c
int main(int argc, char* argv[]) {
    // Check argument count
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <arguments>\n", argv[0]);
        return 1;
    }
    
    // Parse flags
    int flag = 0;
    char* input = NULL;
    
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "-l") == 0) {
            flag = 1;
        } else if(strcmp(argv[i], "-f") == 0 && i+1 < argc) {
            input = argv[++i];
        } else {
            input = argv[i];
        }
    }
    
    return 0;
}
```

### Error Handling Pattern

```c
// Always check return values
int fd = open("file.txt", O_RDONLY);
if(fd < 0) {
    perror("open");  // Print error message
    return 1;
}

// For system calls
if(pthread_create(&tid, NULL, func, NULL) != 0) {
    perror("pthread_create");
    exit(1);
}

// For malloc
void* ptr = malloc(size);
if(!ptr) {
    fprintf(stderr, "malloc failed\n");
    exit(1);
}

// Always free allocated memory
free(ptr);

// Always close file descriptors
close(fd);

// Always destroy synchronization primitives
pthread_mutex_destroy(&lock);
sem_destroy(&sem);
pthread_cond_destroy(&cond);
```

### String Manipulation

```c
#include <string.h>

// Compare strings
if(strcmp(str1, str2) == 0) { /* equal */ }

// Copy string
strcpy(dest, src);
strncpy(dest, src, n);  // Safer, limits length

// Concatenate
strcat(dest, src);
strncat(dest, src, n);

// Find character
char* p = strchr(str, 'x');

// Find substring
char* p = strstr(haystack, needle);

// String length
size_t len = strlen(str);

// Format string
char buffer[100];
sprintf(buffer, "Value: %d", value);
snprintf(buffer, sizeof(buffer), "Safe: %d", value);
```

### Time Functions

```c
#include <time.h>

// Get current time
time_t now = time(NULL);

// Convert to local time
struct tm* tm = localtime(&now);

// Format time
char buf[64];
strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);

// Example fields in struct tm:
tm->tm_year  // Years since 1900
tm->tm_mon   // Month (0-11)
tm->tm_mday  // Day of month (1-31)
tm->tm_hour  // Hour (0-23)
tm->tm_min   // Minute (0-59)
tm->tm_sec   // Second (0-59)
```

---

## 8. Debugging Tips {#debugging-tips}

### Common Race Conditions

#### Counter Race
```c
// WRONG:
int counter = 0;
void* increment() {
    for(int i = 0; i < 1000000; i++)
        counter++;  // NOT ATOMIC!
}

// RIGHT:
pthread_mutex_t lock;
int counter = 0;
void* increment() {
    for(int i = 0; i < 1000000; i++) {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
}
```

#### Buffer Overflow Race
```c
// WRONG:
if(count < MAX) {  // Check
    buffer[count++] = item;  // Use - RACE!
}

// RIGHT:
pthread_mutex_lock(&lock);
if(count < MAX) {
    buffer[count++] = item;
}
pthread_mutex_unlock(&lock);
```

### Common Deadlocks

#### Lock Ordering Deadlock
```c
// WRONG: Thread A and B lock in different order
// Thread A:
pthread_mutex_lock(&lock1);
pthread_mutex_lock(&lock2);

// Thread B:
pthread_mutex_lock(&lock2);  // DEADLOCK!
pthread_mutex_lock(&lock1);

// RIGHT: Always same order
// Both threads:
pthread_mutex_lock(&lock1);
pthread_mutex_lock(&lock2);
```

#### Semaphore Deadlock
```c
// WRONG: Lock held while waiting on semaphore
pthread_mutex_lock(&mutex);
sem_wait(&sem);  // DEADLOCK if sem is 0!
// work
pthread_mutex_unlock(&mutex);

// RIGHT: Release lock before waiting
sem_wait(&sem);
pthread_mutex_lock(&mutex);
// work
pthread_mutex_unlock(&mutex);
```

### Compilation and Execution

#### Compile with pthread
```bash
gcc -pthread program.c -o program
gcc -lpthread program.c -o program  # Alternative
```

#### Run with valgrind (Memory Leaks)
```bash
valgrind --leak-check=full ./program
```

#### Run with helgrind (Thread Errors)
```bash
valgrind --tool=helgrind ./program
```

### Common Errors and Solutions

**Segmentation Fault**
- Dereferencing NULL pointer
- Array out of bounds
- Stack overflow (too much recursion)
- Accessing freed memory

**Race Condition Signs**
- Results vary between runs
- Counter/sum is less than expected
- Intermittent crashes

**Deadlock Signs**
- Program hangs indefinitely
- No progress after certain point
- High CPU but no output

**Memory Leak Signs**
- Memory usage grows over time
- malloc without free
- pthread_create without join
- open without close

### Debugging Techniques

```c
// Add debug prints
#define DEBUG 1
#if DEBUG
#define DPRINTF(fmt, ...) \
    printf("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#else
#define DPRINTF(fmt, ...)
#endif

// Use in code:
DPRINTF("Counter = %d\n", counter);

// Add assertions
#include <assert.h>
assert(count >= 0 && count < MAX);

// Sleep to expose race conditions
#include <unistd.h>
usleep(100);  // Sleep 100 microseconds
sleep(1);     // Sleep 1 second
```

---

## Quick Reference: Function Summary

### Thread Functions
```c
pthread_create(&tid, NULL, func, arg)
pthread_join(tid, &retval)
pthread_exit(retval)
pthread_cancel(tid)
```

### Mutex Functions
```c
pthread_mutex_init(&lock, NULL)
pthread_mutex_lock(&lock)
pthread_mutex_unlock(&lock)
pthread_mutex_destroy(&lock)
```

### Semaphore Functions
```c
sem_init(&sem, 0, initial_value)
sem_wait(&sem)      // P, down, decrement
sem_post(&sem)      // V, up, increment
sem_trywait(&sem)   // Non-blocking
sem_destroy(&sem)
```

### Condition Variable Functions
```c
pthread_cond_init(&cond, NULL)
pthread_cond_wait(&cond, &lock)
pthread_cond_signal(&cond)
pthread_cond_broadcast(&cond)
pthread_cond_destroy(&cond)
```

### File Operations
```c
open(path, flags, mode)
read(fd, buffer, size)
write(fd, buffer, size)
close(fd)
stat(path, &st)
rename(old, new)
unlink(path)
```

### Directory Operations
```c
opendir(path)
readdir(dirp)
closedir(dirp)
```

### Process Functions
```c
fork()
wait(&status)
waitpid(pid, &status, options)
exec...(path, args)
exit(status)
getpid()
getppid()
```

### Signal Functions
```c
signal(signum, handler)
kill(pid, signal)
```

---

## Common Mistakes to Avoid

1. **Forgetting to check return values** - Always check!
2. **Not protecting shared data** - Use mutex/semaphore
3. **Holding lock while waiting** - Release before blocking
4. **Inconsistent lock ordering** - Always same order
5. **Using if instead of while with cond_wait** - Always while!
6. **Forgetting to initialize** - Init mutex/sem before use
7. **Memory leaks** - Free malloc'd memory
8. **Not closing resources** - Close files, destroy locks
9. **Thread argument issues** - Be careful with scope
10. **Busy waiting** - Use proper synchronization

---

## Exam Strategy

1. **Read problem carefully** - Understand requirements
2. **Identify synchronization needs** - What's shared?
3. **Choose right primitive** - Mutex vs semaphore vs cond var
4. **Start with structure** - Data structures, globals
5. **Write init/cleanup** - Initialize everything
6. **Implement thread functions** - One at a time
7. **Add synchronization** - Protect critical sections
8. **Test edge cases** - Empty, full, boundary conditions
9. **Check for deadlocks** - Lock ordering, wait conditions
10. **Comment your code** - Explain synchronization logic

---

## Final Checklist

Before submitting:
- [ ] All threads created and joined
- [ ] All mutexes/semaphores initialized
- [ ] All malloc() have corresponding free()
- [ ] All open() have corresponding close()
- [ ] No busy-waiting loops
- [ ] Proper error checking
- [ ] No race conditions
- [ ] No deadlock possibilities
- [ ] Code compiles without warnings
- [ ] Tested with multiple runs

---

**Good luck on your exam!**