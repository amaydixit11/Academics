#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>


#define NUM_THREADS 8
#define MAX_CONCURRENT 3     // maximum threads allowed in critical section


sem_t multiplex;


void* worker(void* arg) 
{
  int id = *(int*)arg;
  // Request to enter
  sem_wait(&multiplex); // take one token
  printf("Thread %d ENTERED critical section.\n", id);
  // Critical section (simulate some work)
  sleep(1); // simulate processing
  printf("Thread %d LEAVING critical section.\n", id);
  sem_post(&multiplex); // release token
  return NULL;
}


int main() 
{
  pthread_t threads[NUM_THREADS];
  int ids[NUM_THREADS];
  
  // Initialize semaphore to MAX_CONCURRENT
  sem_init(&multiplex, 0, MAX_CONCURRENT);
  
  // Create threads
  for (int i = 0; i < NUM_THREADS; i++) 
  {
    ids[i] = i + 1;
    pthread_create(&threads[i], NULL, worker, &ids[i]);
  }
  
  // Join threads
  for (int i = 0; i < NUM_THREADS; i++) 
  {
    pthread_join(threads[i], NULL);
  }
  
  sem_destroy(&multiplex);
  printf("All threads finished.\n");
  return 0;
}
