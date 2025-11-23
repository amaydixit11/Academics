#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t sem;

void* threadA(void* arg) 
{
  printf("Hello from A\n");
  sem_post(&sem);
  return NULL;
}
void* threadB(void* arg) 
{
  sem_wait(&sem);
  printf("Hello from B\n");
  return NULL;
}
int main()
{
  pthread_t tA, tB;
  
  sem_init(&sem, 0, 0);
  
  pthread_create(&tA, NULL, threadA, NULL);
  pthread_create(&tB, NULL, threadB, NULL);
  
  pthread_join(tA, NULL);
  pthread_join(tB, NULL);
  
  sem_destroy(&sem);
  return 0;
}
