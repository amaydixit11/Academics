#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LOGS 10

char *log_buffer[MAX_LOGS];
int count = 0;

// ADD YOUR CODE HERE

void *worker(void *id) {
    for (int i = 0; i < 3; i++) {
        char msg[64];
        sprintf(msg, "Worker %ld message %d", (long)id, i);
        // ADD YOUR CODE HERE
        log_buffer[count++] = strdup(msg);
        printf("Worker %ld queued log. (count=%d)\n", (long)id, count);
        // ADD YOUR CODE HERE
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
        // ADD YOUR CODE HERE
        char *msg = log_buffer[--count];
        // ADD YOUR CODE HERE
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
