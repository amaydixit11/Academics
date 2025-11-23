#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

int m, n;
int *AVAILABLE, **ALLOC, **NEED, *REQ, reqfrom;
char reqtype;
pthread_mutex_t rmtx = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t pmtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t *cwait;
pthread_mutex_t *cmtx;
pthread_barrier_t BOS, REQB, *ACKB;

typedef struct _node {
  int from;
  int *request;
  struct _node *next;
} node;

typedef struct {
   node *F;
   node *B;
} RQ;

RQ initQ ( )
{
   RQ Q;

   Q.F = (node *)malloc(sizeof(node));
   Q.F -> from = -1;
   Q.F -> request = NULL;
   Q.F -> request = NULL;
   Q.B = NULL;
   return Q;
}

int emptyQ ( RQ Q )
{
   return (Q.B == NULL);
}

node *frontQ ( RQ Q )
{
   if (emptyQ(Q)) return NULL;
   return Q.F -> next;
}

RQ enQ ( RQ Q, int i )
{
   node *p;
   int j;

   p = (node *)malloc(sizeof(node));
   p -> from = i;
   p -> request = (int *)malloc(m * sizeof(int));
   for (j=0; j<m; ++j) (p -> request)[j] = REQ[j];
   p -> next = NULL;
   if (Q.B == NULL) {
      Q.F -> next = Q.B = p;
   } else {
      Q.B -> next = p;
      Q.B = p;
   }
   return Q;
}

RQ deQ ( RQ Q )
{
   node *p;

   if (emptyQ(Q)) return Q;
   p = Q.F -> next;
   Q.F -> next = Q.F -> next -> next;
   if (Q.F -> next == NULL) Q.B = NULL;
   free(p -> request);
   free(p);
   return Q;
}

void prnQ ( RQ Q )
{
   node *p;

   printf("\t\tWaiting threads:");
   p = Q.F -> next;
   while (p) {
      printf(" %d", p -> from);
      p = p -> next;
   }
   printf("\n");
}

void timed_wait ( int t )
{
   usleep(t * 50000);
}

void *tmain ( void *arg )
{
   int i, j, wtime, flag;
   char fname[64], rt[32];
   FILE *fp;

   i = *((int *)arg);
   printf("\tThread %d born\n", i);
   sprintf(fname, "input/thread%02d.txt", i);
   fp = (FILE *)fopen(fname, "r");
   for (j=0; j<m; ++j) fscanf(fp, "%d", &NEED[i][j]);
   pthread_barrier_wait(&BOS);

   while (1) {
      fscanf(fp, "%d%s", &wtime, rt);
      timed_wait(wtime);
      if (rt[0] == 'Q') {
         pthread_mutex_lock(&rmtx);
         reqfrom = i;
         reqtype = 'Q';
         pthread_barrier_wait(&REQB);
         pthread_barrier_wait(ACKB+i);
         pthread_mutex_unlock(&rmtx);
         pthread_mutex_lock(&pmtx);
         printf("\tThread %d going to quit\n", i);
         pthread_mutex_unlock(&pmtx);
         break;
      } else if (rt[0] == 'R') {
         pthread_mutex_lock(&rmtx);
         pthread_mutex_lock(cmtx+i);
         reqfrom = i;
         reqtype = 'R';
         flag = 0;
         for (j=0; j<m; ++j) {
            fscanf(fp, "%d", &REQ[j]);
            if (REQ[j] > 0) flag = 1;
         }
         pthread_mutex_lock(&pmtx);
         printf("\tThread %d sends resource request: type = %s\n",
                i, (flag) ? "ADDITIONAL" : "RELEASE");
         pthread_mutex_unlock(&pmtx);
         pthread_barrier_wait(&REQB);
         pthread_barrier_wait(ACKB+i);
         pthread_mutex_unlock(&rmtx);

         if (flag) pthread_cond_wait(cwait+i,cmtx+i);
         pthread_mutex_unlock(cmtx+i);
         pthread_mutex_lock(&pmtx);
         if (flag)
            printf("\tThread %d is granted its last resource request\n", i);
         else
            printf("\tThread %d is done with its resource release request\n", i);
         pthread_mutex_unlock(&pmtx);
      }
   }
   fclose(fp);
   pthread_exit(NULL);
}

void initsession ( )
{
   int i, j;
   FILE *fp;

   fp = (FILE *)fopen("input/system.txt","r");
   fscanf(fp, "%d%d", &m, &n);

   AVAILABLE = (int *)malloc(m * sizeof(int));
   for (j=0; j<m; ++j) fscanf(fp, "%d", AVAILABLE+j);
   fclose(fp);

   ALLOC = (int **)malloc(n * sizeof(int *));
   NEED = (int **)malloc(n * sizeof(int *));
   for (i=0; i<n; ++i) {
      ALLOC[i] = (int *)malloc(m * sizeof(int));
      NEED[i] = (int *)malloc(m * sizeof(int));
      for (j=0; j<m; ++j) ALLOC[i][j] = NEED[i][j] = 0;
   }
   REQ = (int *)malloc(m * sizeof(int));
   cwait = (pthread_cond_t *)malloc(n * sizeof(pthread_cond_t));
   cmtx = (pthread_mutex_t *)malloc(n * sizeof(pthread_mutex_t));
   for (i=0; i<n; ++i) {
      cwait[i] = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
      cmtx[i] = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
   }
   ACKB = (pthread_barrier_t *)malloc(n * sizeof(pthread_barrier_t));
   pthread_barrier_init(&REQB, NULL, 2);
   for (i=0; i<n; ++i) pthread_barrier_init(ACKB+i,NULL,2);

   pthread_barrier_init(&BOS,NULL,n+1);
}

pthread_t *createthreads ( )
{
   int i, *targ;
   pthread_t *tid;

   tid = (pthread_t *)malloc(n * sizeof(pthread_t));
   targ = (int *)malloc(n * sizeof(int));
   for (i=0; i<n; ++i) {
      targ[i] = i;
      pthread_create(tid+i, NULL, tmain, (void *)(targ + i));
   }
   free(targ);
   return tid;
}

int registerrequest ( int i )
{
   int j, flag;

   flag = 0;
   for (j=0; j<m; ++j) {
      if (REQ[j] < 0) {
         ALLOC[i][j] += REQ[j];
         NEED[i][j] -= REQ[j];
         AVAILABLE[j] -= REQ[j];
         REQ[j] = 0;
      } else if (REQ[j] > 0) {
         flag = 1;
      }
   }
   return flag;
}

int resourceavailable ( node *q )
{
   int j, flag;

   flag = 1;
   for (j=0; j<m; ++j) {
      if ((q -> request)[j] > AVAILABLE[j]) {
         flag = 0;
         break;
      }
   }
   pthread_mutex_lock(&pmtx);
   if (flag == 0)
      printf("    +++ Insufficient resources to grant request of thread %d\n", q -> from);
   pthread_mutex_unlock(&pmtx);
   return flag;
}

int issafe ( int *exited )
{
   int *WORK, *FINISH, i, j, flag;

   WORK = (int *)malloc(m * sizeof(int));
   for (j=0; j<m; ++j) WORK[j] = AVAILABLE[j];

   FINISH = (int *)malloc(n * sizeof(int));
   for (i=0; i<n; ++i) FINISH[i] = exited[i];

   while (1) {
      flag = 0;
      for (i=0; i<n; ++i) {
         if (!FINISH[i]) {
            for (j=0; j<m; ++j) if (NEED[i][j] > WORK[j]) break;
            if (j == m) {
               for (j=0; j<m; ++j) WORK[j] += ALLOC[i][j];
               FINISH[i] = 1;
               flag = 1;
            }
         }
         if (flag) break;
      }
      if (!flag) break;
   }

   flag = 1;
   for (i=0; i<n; ++i) {
      if (!FINISH[i]) {
         flag = 0;
         break;
      }
   }

   free(WORK);
   free(FINISH);

   return flag;
}

int safetoallocate ( node *q, int *exited )
{
   int i, j, s, *REQ;

   i = q -> from;
   REQ = q -> request;
   for (j=0; j<m; ++j) {
      ALLOC[i][j] += REQ[j];
      NEED[i][j] -= REQ[j];
      AVAILABLE[j] -= REQ[j];
   }

   s = issafe(exited);

   for (j=0; j<m; ++j) {
      ALLOC[i][j] -= REQ[j];
      NEED[i][j] += REQ[j];
      AVAILABLE[j] += REQ[j];
   }

   if (!s) {
      pthread_mutex_lock(&pmtx);
      printf("    +++ Unsafe to grant request of thread %d\n", i);
      pthread_mutex_unlock(&pmtx);
   }

   return s;
}

int canserve ( node *q, int *exited )
{
   #ifdef _DLAVOID
      return (resourceavailable(q) && safetoallocate(q,exited));
   #else
      return resourceavailable(q);
   #endif
}

RQ serverequest ( RQ Q, int *exited )
{
   node *p, *q;
   int i, j, req;

   if (emptyQ(Q)) return Q;
   p = Q.F;
   q = p -> next;
   while (q) {
      if (canserve(q,exited)) {
         i = q -> from;
         for (j=0; j<m; ++j) {
            req = (q -> request)[j];
            if (req) {
               ALLOC[i][j] += req;
               NEED[i][j] -= req;
               AVAILABLE[j] -= req;
            }
         }
         p -> next = q -> next;
         free(q -> request);
         free(q);
         q = p -> next;
         pthread_mutex_lock(&pmtx);
         printf("Master thread grants resource request for thread %d\n", i);
         pthread_mutex_unlock(&pmtx);
         pthread_mutex_lock(cmtx+i);
         pthread_cond_signal(cwait+i);
         pthread_mutex_unlock(cmtx+i);
      } else {
         p = q;
         q = q -> next;
      }
   }
   if (Q.F -> next == NULL) {
      Q.B = NULL;
   } else {
      Q.B = Q.F -> next;
      while (Q.B -> next != NULL) Q.B = Q.B -> next;
   }
   return Q;
}

void releaseresources ( int i )
{
   int j;

   for (j=0; j<m; ++j) {
      AVAILABLE[j] += ALLOC[i][j];
      ALLOC[i][j] = 0;
      NEED[i][j] = 0;
   }
}

int main ()
{
   int i, N, *exited;
   char rt;
   RQ Q = initQ();

   initsession();
   createthreads();
   pthread_barrier_wait(&BOS);

   N = n;
   exited = (int *)malloc(n * sizeof(int));
   for (i=0; i<n; ++i) exited[i] = 0;

   while (1) {
      pthread_barrier_wait(&REQB);
      i = reqfrom;
      rt = reqtype;
      if (rt == 'R') {
         if (registerrequest(i)) {
            Q = enQ(Q,i);
            pthread_mutex_lock(&pmtx);
            printf("Master thread stores resource request of thread %d\n", i);
            prnQ(Q);
            pthread_mutex_unlock(&pmtx);
         }
         pthread_barrier_wait(ACKB+i);
      } else if (rt == 'Q') {
         --N;
         exited[i] = 1;
         releaseresources(i);
         pthread_barrier_wait(ACKB+i);
         pthread_mutex_lock(&pmtx);
         printf("Master thread releases resources of thread %d\n", i);
         prnQ(Q); fflush(stdout);
         printf("%d threads left:", N);
         for (i=0; i<n; ++i) if (!exited[i]) printf(" %d", i);
         printf("\n");
         printf("Available resources:");
         for (i=0; i<m; ++i) printf(" %d", AVAILABLE[i]);
         printf("\n");
         pthread_mutex_unlock(&pmtx);
         if (N == 0) break;
      } else {
         fprintf(stderr, "Master thread finds unknown request from thread %d\n", i);
         exit(1);
      }

      pthread_mutex_lock(&pmtx);
      printf("Master thread tries to grant pending requests\n");
      pthread_mutex_unlock(&pmtx);
      Q = serverequest(Q,exited);
      pthread_mutex_lock(&pmtx);
      prnQ(Q);
      pthread_mutex_unlock(&pmtx);
   }

   exit(0);
}
