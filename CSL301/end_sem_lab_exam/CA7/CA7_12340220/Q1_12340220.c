    #include <stdio.h>
    #include <pthread.h>

    int global = 10;

    void* threadFunction(void* arg) {
        int id = *(int*)arg;     
        int local_thread = id * 100;

        printf("Thread %d:\n", id);
        printf("    global address: %p\n", (void*)&global);
        printf("    local_thread  address: %p\n\n", (void*)&local_thread);

        return NULL;
    }

    int main() {
        pthread_t t1, t2;
        int id1 = 1, id2 = 2;
        int local_main = 20;

        printf("Main thread:\n");
        printf("    global address: %p\n", (void*)&global);
        printf("    local_main address: %p\n\n", (void*)&local_main);

        pthread_create(&t1, NULL, threadFunction, &id1);
        pthread_create(&t2, NULL, threadFunction, &id2);

        pthread_join(t1, NULL);
        pthread_join(t2, NULL);

        return 0;
    }
