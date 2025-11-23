#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t semA, semB;

void* threadA(void* arg) {
  printf("Hello from A\n");
  sem_post(&semA);
  return NULL;
}

void* threadB(void* arg) {
  sem_wait(&semA);
  printf("Hello from B\n");
  sem_post(&semB);
  return NULL;
}

void* threadC(void* arg) {
  sem_wait(&semB);
  printf("Hello from C\n");
  return NULL;
}

int main() {
  pthread_t tA, tB, tC;

  sem_init(&semA, 0, 0);
  sem_init(&semB, 0, 0);

  pthread_create(&tA, NULL, threadA, NULL);
  pthread_create(&tB, NULL, threadB, NULL);
  pthread_create(&tC, NULL, threadC, NULL);

  pthread_join(tA, NULL);
  pthread_join(tB, NULL);
  pthread_join(tC, NULL);

  sem_destroy(&semA);
  sem_destroy(&semB);

  return 0;
}
