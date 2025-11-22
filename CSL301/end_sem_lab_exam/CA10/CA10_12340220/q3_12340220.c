#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LOGS 10

char *log_buffer[MAX_LOGS];
int count = 0;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t can_log = PTHREAD_COND_INITIALIZER;
pthread_cond_t can_add = PTHREAD_COND_INITIALIZER;

void *worker(void *id) {
    for (int i = 0; i < 3; i++) {
        char msg[64];
        sprintf(msg, "Worker %ld message %d", (long)id, i);

        pthread_mutex_lock(&lock);
        while (count == MAX_LOGS)  // buffer full
            pthread_cond_wait(&can_add, &lock);

        log_buffer[count++] = strdup(msg);
        printf("Worker %ld queued log. (count=%d)\n", (long)id, count);

        pthread_cond_signal(&can_log);  // notify logger
        pthread_mutex_unlock(&lock);

        usleep(100000);
    }
    return NULL;
}

void *logger(void *arg) {
    FILE *f = fopen("log.txt", "w");
    if (!f) {
        perror("fopen");
        return NULL;
    }

    while (1) {
        pthread_mutex_lock(&lock);
        while (count == 0)  // nothing to log
            pthread_cond_wait(&can_log, &lock);

        char *msg = log_buffer[--count];
        pthread_cond_signal(&can_add);  // space available
        pthread_mutex_unlock(&lock);

        fprintf(f, "%s\n", msg);
        fflush(f);
        printf("Logger wrote: %s\n", msg);

        free(msg);
        usleep(50000);
    }

    fclose(f);
    return NULL;
}


int main() {
    pthread_t log_thread, workers[3];
    pthread_create(&log_thread, NULL, logger, NULL);
    for (long i = 0; i < 3; i++)
        pthread_create(&workers[i], NULL, worker, (void *)i);

    for (int i = 0; i < 3; i++)
        pthread_join(workers[i], NULL);

    sleep(1);

    pthread_cancel(log_thread);
    pthread_join(log_thread, NULL);

    printf("\nAll workers finished. Check 'log.txt' for output.\n");
    return 0;
}
