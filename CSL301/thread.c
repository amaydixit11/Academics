#include <pthread.h>
#include <stdio.h>

void* printid(void* arg){
    int id = *(int*) arg;
    printf("Thread ID: %d\n", id);
    return NULL;
}

void* myfunc(void* arg){
    printf("Hello from Threak\n");
    return NULL;
}

int main(){
    pthread_t tid;
    int id = 12340220;
    pthread_create(&tid, NULL, printid, &id);
    pthread_join(tid, NULL);
    printf("Thread finished!\n");
    return 0;
}