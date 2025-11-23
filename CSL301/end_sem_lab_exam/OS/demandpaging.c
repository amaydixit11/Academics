#include <stdio.h>
#include <stdlib.h>

#define NFRAMES 12288
#define PTSIZE 2048

typedef unsigned short int ptentry;

typedef struct _node {
   int item;
   struct _node *next;
} node;

typedef struct {
   node *F;
   node *B;
} queue;

typedef struct {
   int n;
   int m;
   int *size;
   int **sidx;
   int *scnt;
} simdata;

typedef struct {
   int npageaccess;
   int npagefault;
   int nswap;
   int nactive;
   int domp;
   ptentry **pagetable;
   queue freeframelist;
   queue readylist;
   queue swaplist;
} kerneldata;

queue initQ ( )
{
   return (queue){NULL, NULL};
}

int emptyQ ( queue Q )
{
   return (Q.F == NULL);
}

int front ( queue Q )
{
   if (Q.F == NULL) return -1;
   return Q.F -> item;
}

queue enQ ( queue Q, int x )
{
   node *p;

   p = (node *)malloc(sizeof(node));
   p -> item = x;
   p -> next = NULL;
   if (Q.F == NULL) {
      Q.F = Q.B = p;
   } else {
      Q.B -> next = p;
      Q.B = p;
   }
   return Q;
}

queue deQ ( queue Q )
{
   node *p;

   if (Q.F == NULL) return Q;
   p = Q.F;
   Q.F = Q.F -> next;
   if (Q.F == NULL) Q.B = NULL;
   free(p);
   return Q;
}

simdata readsimdata ( )
{
   FILE *fp;
   int n, m;
   simdata S;
   int i, j;

   fp = (FILE *)fopen("search.txt", "r");
   fscanf(fp, "%d%d", &n, &m);
   S.n = n;
   S.m = m;
   S.size = (int *)malloc(n * sizeof(int));
   S.sidx = (int **)malloc(n * sizeof(int *));
   S.scnt = (int *)malloc(n * sizeof(int));
   for (i=0; i<n; ++i) {
      fscanf(fp, "%d", S.size + i);
      S.sidx[i] = (int *)malloc(m * sizeof(int));
      for (j=0; j<m; ++j) fscanf(fp, "%d", S.sidx[i]+j);
      S.scnt[i] = 0;
   }
   fclose(fp);
   return S;
}

kerneldata loadprocess ( kerneldata KD, int i )
{
   int j;

   for (j=0; j<10; ++j) {
      KD.pagetable[i][j] = front(KD.freeframelist) | (1U << 15);
      KD.freeframelist = deQ(KD.freeframelist);
   }
   for (j=10; j<PTSIZE; ++j) KD.pagetable[i][j] = 0;
   return KD;
}

kerneldata initkerneldata ( simdata SD )
{
   kerneldata KD;
   int i, n;

   n = SD.n;

   KD.npageaccess = KD.npagefault = KD.nswap = 0;
   KD.freeframelist = initQ();
   KD.readylist = initQ();
   KD.swaplist = initQ();

   for (i=0; i<NFRAMES; ++i) KD.freeframelist = enQ(KD.freeframelist,i);

   KD.pagetable = (ptentry **)malloc(n * sizeof(ptentry *));
   for (i=0; i<n; ++i) {
      KD.pagetable[i] = (ptentry *)malloc(PTSIZE * sizeof(ptentry));
      KD = loadprocess(KD,i);
      KD.readylist = enQ(KD.readylist, i);
   }

   KD.nactive = KD.domp = n;

   return KD;
}

kerneldata unloadprocess ( kerneldata KD, int i )
{
   int p;

   for (p=0; p<PTSIZE; ++p) {
      if (KD.pagetable[i][p] & (1U << 15)) {
         KD.freeframelist = enQ(KD.freeframelist, p ^ (1U << 15));
      }
   }
   return KD;
}

kerneldata bsloop ( simdata SD, kerneldata KD, int i, int k, int size )
{
   int L, R, M, j;
   int p;

   L = 0; R = size - 1;
   while (L < R) {
      M = (L + R) / 2;
      ++(KD.npageaccess);
      p = (M >> 10);
      if ((KD.pagetable[i][p+10] & (1U << 15)) == 0) {
         ++(KD.npagefault);
         if (!emptyQ(KD.freeframelist)) {
            KD.pagetable[i][p+10] = front(KD.freeframelist) | (1U << 15);
            KD.freeframelist = deQ(KD.freeframelist);
         } else {
            ++(KD.nswap);
            --(KD.nactive);
            if (KD.nactive < KD.domp) KD.domp = KD.nactive;
            printf("+++ Swapping out process %4d  [%d active processes]\n", i, KD.nactive);
            KD.swaplist = enQ(KD.swaplist, i);
            KD = unloadprocess(KD, i);
            return KD;
         }
      }
      if (k <= M) R = M;
      else L = M + 1;
   }

   ++(SD.scnt[i]);
   if (SD.scnt[i] < SD.m) {
      KD.readylist = enQ(KD.readylist, i);
   } else {
      KD = unloadprocess(KD, i);
      --(KD.nactive);
      if (!emptyQ(KD.swaplist)) {
         i = front(KD.swaplist);
         KD.swaplist = deQ(KD.swaplist);
         KD = loadprocess(KD,i);
         ++(KD.nactive);
         printf("+++ Swapping in process  %4d  [%d active processes]\n", i, KD.nactive);
         j = SD.scnt[i];
         k = SD.sidx[i][j];
         #ifdef VERBOSE
         printf("\tSearch %d by Process %d\n", j+1, i);
         #endif
         KD = bsloop(SD, KD, i, k, SD.size[i]);
      }
   }
   return KD;
}

kerneldata nextsearch ( simdata SD, kerneldata KD )
{
   int i, j, k;

   i = front(KD.readylist);
   KD.readylist = deQ(KD.readylist);
   j = SD.scnt[i];
   k = SD.sidx[i][j];
   #ifdef VERBOSE
   printf("\tSearch %d by Process %d\n", j+1, i);
   #endif
   KD = bsloop(SD, KD, i, k, SD.size[i]);
   
   return KD;
}

void printstat ( kerneldata KD )
{
   printf("+++ Page access summary\n");
   printf("\tTotal number of page accesses  =  %d\n", KD.npageaccess);
   printf("\tTotal number of page faults    =  %d\n", KD.npagefault);
   printf("\tTotal number of swaps          =  %d\n", KD.nswap);
   printf("\tDegree of multiprogramming     =  %d\n", KD.domp);
}

int main ()
{
   simdata SD;
   kerneldata KD;

   SD = readsimdata();
   printf("+++ Simulation data read from file\n");

   KD = initkerneldata(SD);
   printf("+++ Kernel data initialized\n");


   while (!emptyQ(KD.readylist)) KD = nextsearch(SD,KD);

   printstat(KD);

   exit(0);
}
