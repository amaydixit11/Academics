#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_JOBS 5
#define NUM_WORKERS 3

int jobs[MAX_JOBS];
int count = 0; 
// ADD YOUR CODE

void *dispatcher(void *arg) {
    int job_id = 1;
    while (job_id <= 10) {
        pthread_mutex_lock(&lock);

        // ADD YOUR CODE

        jobs[count++] = job_id;
        printf("Dispatcher added job %d (count=%d)\n", job_id, count);

        pthread_mutex_unlock(&lock);
        job_id++;

        usleep(100000); 
    }
    return NULL;
}

void *worker(void *arg) {
    long id = (long)arg;

    while (1) {
        pthread_mutex_lock(&lock);

       // ADD YOUR CODE

        int job = jobs[--count];
        printf("Worker %ld processing job %d (remaining=%d)\n", id, job, count);

        // ADD YOUR CODE

        pthread_mutex_unlock(&lock);

        usleep(200000); 
    }
    return NULL;
}

int main() {
    pthread_t disp, workers[NUM_WORKERS];

    pthread_create(&disp, NULL, dispatcher, NULL);
    for (long i = 0; i < NUM_WORKERS; i++)
        pthread_create(&workers[i], NULL, worker, (void *)i);

    pthread_join(disp, NULL);
    sleep(2); 

    printf("All jobs dispatched. Exiting...\n");
    return 0;
}
