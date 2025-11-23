#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#define MAX_SIZE 256

int b, r, n;

typedef struct {
   int value;
   pthread_mutex_t mtx;
   pthread_cond_t cv;
} semaphore;

semaphore boat = {0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
semaphore rider = {0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};

pthread_mutex_t bmtx = PTHREAD_MUTEX_INITIALIZER;

pthread_barrier_t EOS, BB[MAX_SIZE];

int BA[MAX_SIZE], BC[MAX_SIZE], BT[MAX_SIZE];

void P ( semaphore *s )
{
   pthread_mutex_lock(&(s -> mtx));
   --(s -> value);
   if (s -> value < 0) pthread_cond_wait(&(s -> cv), &(s -> mtx));
   pthread_mutex_unlock(&(s -> mtx));
}

void V ( semaphore *s )
{
   pthread_mutex_lock(&(s -> mtx));
   ++(s -> value);
   if (s -> value <= 0) pthread_cond_signal(&(s -> cv));
   pthread_mutex_unlock(&(s -> mtx));
}

void delall ( )
{
   int i;

   pthread_mutex_destroy(&bmtx);
   pthread_mutex_destroy(&(boat.mtx));
   pthread_mutex_destroy(&(rider.mtx));
   pthread_cond_destroy(&(boat.cv));
   pthread_cond_destroy(&(rider.cv));
   for (i=1; i<=b; ++i) pthread_barrier_destroy(&BB[i]);
   pthread_barrier_destroy(&EOS);
}

void *bmain ( void *barg )
{
   int bno, rno, rtime;

   bno = *(int *)barg;

   BA[bno] = 1; BC[bno] = -1;
   pthread_barrier_init(&BB[bno], NULL, 2);

   printf("Boat    %4d    Ready\n", bno);

   while (1) {
      V(&rider);
      P(&boat);

      pthread_barrier_wait(&BB[bno]);

      pthread_mutex_lock(&bmtx);
      rno = BC[bno];
      BA[bno] = 0;
      pthread_mutex_unlock(&bmtx);

      printf("Boat    %4d    Start of ride for visitor %2d\n", bno, rno);
      rtime = BT[bno];
      usleep(rtime * 100000);
      printf("Boat    %4d    End of ride for visitor %2d (ride time = %d)\n", bno, rno, rtime);

      pthread_mutex_lock(&bmtx);
      BA[bno] = 1; BC[bno] = -1;
      --n;
      if (n == 0) {
         pthread_mutex_unlock(&bmtx);
         pthread_barrier_wait(&EOS);
      } else {
         pthread_barrier_init(&BB[bno], NULL, 2);
         pthread_mutex_unlock(&bmtx);
      }
   }

   pthread_exit(NULL);
}

void *rmain ( void *rarg )
{
   int rno, vtime, bno, i, rtime;

   rno = *(int *)rarg;

   vtime = 30 + rand() % 91;
   rtime = 15 + rand() % 46;
   printf("Visitor %4d    Starts sightseeing for %3d minutes\n", rno, vtime);
   usleep(vtime * 100000);
   printf("Visitor %4d    Ready to ride a boat (ride time = %d)\n", rno, rtime);

   V(&boat);
   P(&rider);

   bno = -1;
   while (bno == -1) {
      pthread_mutex_lock(&bmtx);
      for (i=1; i<=b; ++i) {
         if ((BA[i] == 1) && (BC[i] == -1)) {
            bno = i;
            BC[bno] = rno;
            BT[bno] = rtime;
            break;
         }
      }
      pthread_mutex_unlock(&bmtx);
   }
   printf("Visitor %4d    Finds boat %2d\n", rno, bno);
   pthread_barrier_wait(&BB[bno]);

   usleep(rtime * 100000);

   printf("Visitor %4d    Leaving\n", rno);

   pthread_exit(NULL);
}

int main ( int argc, char *argv[] )
{
   int i, targ[MAX_SIZE];
   pthread_t bthread[MAX_SIZE], rthread[MAX_SIZE];

   srand((unsigned int)time(NULL));

   for (i=0; i<MAX_SIZE; ++i) targ[i] = i+1;

   if (argc < 3) {
      fprintf(stderr, "Run with number of boats and number of riders\n");
      exit(1);
   }

   b = atoi(argv[1]);
   n = r = atoi(argv[2]);

   pthread_barrier_init(&EOS, NULL, 2);

   for (i=0; i<b; ++i) pthread_create(bthread+i, NULL, bmain, (void *)(targ+i));
   for (i=0; i<r; ++i) pthread_create(rthread+i, NULL, rmain, (void *)(targ+i));

   pthread_barrier_wait(&EOS);

   for (i=0; i<b; ++i) pthread_cancel(bthread[i]);
   for (i=0; i<r; ++i) pthread_cancel(rthread[i]);

   delall();

   exit(0);
}
