#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t can_read = PTHREAD_COND_INITIALIZER;
pthread_cond_t can_write = PTHREAD_COND_INITIALIZER;

int read_count = 0;
int write_count = 0;
int waiting_writers = 0;

void start_read() {
    pthread_mutex_lock(&lock);
    while (/* ADD YOUR CODE HERE */) {
        pthread_cond_wait(&can_read, &lock);
    }
    read_count++;
    pthread_mutex_unlock(&lock);
}

void end_read() {
    pthread_mutex_lock(&lock);
    read_count--;
    if (read_count == 0)
        pthread_cond_signal(&can_write);
    pthread_mutex_unlock(&lock);
}

void start_write() {
    pthread_mutex_lock(&lock);
    waiting_writers++;
    while (/* ADD YOUR CODE HERE */) {
        pthread_cond_wait(&can_write, &lock);
    }
    waiting_writers--;
    write_count = 1;
    pthread_mutex_unlock(&lock);
}

void end_write() {
    pthread_mutex_lock(&lock);
    write_count = 0;
    // ADD YOUR CODE HERE
    pthread_mutex_unlock(&lock);
}

void *reader(void *id) {
    for (int i = 0; i < 5; i++) {
        start_read();
        printf("Reader %ld reading\n", (long)id);
        usleep(100000);
        end_read();
    }
    return NULL;
}

void *writer(void *id) {
    for (int i = 0; i < 3; i++) {
        start_write();
        printf("Writer %ld writing\n", (long)id);
        usleep(150000);
        end_write();
    }
    return NULL;
}

int main() {
    pthread_t r[3], w[2];
    for (long i = 0; i < 3; i++) pthread_create(&r[i], NULL, reader, (void *)i);
    for (long i = 0; i < 2; i++) pthread_create(&w[i], NULL, writer, (void *)i);
    for (int i = 0; i < 3; i++) pthread_join(r[i], NULL);
    for (int i = 0; i < 2; i++) pthread_join(w[i], NULL);
    return 0;
}

