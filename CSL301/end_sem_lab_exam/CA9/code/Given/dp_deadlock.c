#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define N 5  

sem_t forks[N];  
int eat_count[N] = {0};  

void* philosopher(void* arg) {
    int id = *(int*)arg;
    int left_fork = id;
    int right_fork = (id + 1) % N;
    
    for(int i = 0; i < 3; i++) {
        
        printf("Philosopher %d is thinking...\n", id);
        usleep(100000);
        
       
        printf("Philosopher %d is hungry, reaching for forks %d and %d\n", 
               id, left_fork, right_fork);
        
        sem_wait(&forks[left_fork]);  
        printf("  Philosopher %d picked up left fork %d\n", id, left_fork);
        usleep(50000);  
        
        sem_wait(&forks[right_fork]); 
        printf("  Philosopher %d picked up right fork %d\n", id, right_fork);
        
       
        printf("Philosopher %d is EATING (meal #%d)\n", id, eat_count[id] + 1);
        eat_count[id]++;
        usleep(200000);
        
        
        sem_post(&forks[left_fork]);
        sem_post(&forks[right_fork]);
        printf("  Philosopher %d put down both forks\n\n", id);
    }
    
    return NULL;
}

int main() {
    pthread_t philosophers[N];
    int ids[N];
    
   
    for(int i = 0; i < N; i++) {
        sem_init(&forks[i], 0, 1);
        ids[i] = i;
    }
    
    printf("=== DINING PHILOSOPHERS - DEADLOCK VERSION ===\n");
    printf("Number of philosophers: %d\n", N);
    printf("Watch for deadlock...\n\n");
    
    
    for(int i = 0; i < N; i++) {
        pthread_create(&philosophers[i], NULL, philosopher, &ids[i]);
    }
    
   
    for(int i = 0; i < N; i++) {
        pthread_join(philosophers[i], NULL);
    }
    
    
    for(int i = 0; i < N; i++) {
        sem_destroy(&forks[i]);
    }
    
    printf("\n========== MEAL COUNT ==========\n");
    for(int i = 0; i < N; i++) {
        printf("Philosopher %d ate %d times (expected: 3)\n", i, eat_count[i]);
    }
    
    return 0;
}

