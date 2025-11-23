#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#define MAX_NAME_SIZE 16384
#define MAX_USER_CNT 16384

typedef struct {
   unsigned uid;
   char *uname;
} user;

int getusernames ( user *UNAME )
{
   FILE *fp;
   int n;
   char line[8192], *p, *q;

   fp = (FILE *)fopen("/etc/passwd", "r");
   if (fp == NULL) {
      fprintf(stderr, "*** Unable to open /etc/passwd\n");
      exit(1);
   }
   while (1) {
      fgets(line, 8192, fp);
      if (feof(fp)) break;
      p = strchr(line,':');
      *p = '\0';
      UNAME[n].uname = (char *)malloc((strlen(line) + 1) * sizeof(char));
      strcpy(UNAME[n].uname, line);
      *p = ':';
      q = strchr(p+1,':');
      p = strchr(q+1,':');
      *p = '\0';
      UNAME[n].uid = atoi(q+1);
      *p = ':';
      ++n;
      if (n == MAX_USER_CNT) break;
   }
   fclose(fp);
   return n;
}

void printusername ( unsigned uid, user *UNAME, int u )
{
   int i;

   for (i=0; i<u; ++i) {
      if (uid == UNAME[i].uid) {
         printf("%-20s", UNAME[i].uname);
         return;
      }
   }
   printf("UNKNOWN             ");
}

int findall ( const char sroot[], const char extn[], int n, user *UNAME, int u )
{
   DIR *dp;
   struct dirent *entry;
   char subdir[MAX_NAME_SIZE], fname[MAX_NAME_SIZE], *p;
   struct stat fcb;

   dp = (DIR *)opendir(sroot);
   if (dp == NULL) {
      fprintf(stderr, "*** Unable to access %s\n", sroot);
      return n;
   }

   while (1) {
      entry = readdir(dp);
      if (entry == NULL) {
         closedir(dp);
         return n;
      }
      if (!strcmp(entry -> d_name, ".") || !strcmp(entry -> d_name, "..")) continue;
      if (entry -> d_type == DT_DIR) {
         sprintf(subdir, "%s/%s", sroot, entry -> d_name);
         n = findall(subdir, extn, n, UNAME, u);
      } else if (entry -> d_type == DT_REG) {
         p = (entry -> d_name) + strlen(entry -> d_name) - strlen(extn);
         if (!strcmp(p,extn)) {
            ++n;
            sprintf(fname, "%s/%s", sroot, entry -> d_name);
            stat(fname, &fcb);
            printf("%-10d: ", n); printusername(fcb.st_uid,UNAME,u);
            printf("%-20ld %s\n", fcb.st_size, fname);
         }
      }
   }
}

int main ( int argc, char *argv[] )
{
   int n, u;
   char *extn;
   user UNAME[MAX_USER_CNT];

   if (argc < 3) {
      fprintf(stderr, "*** Run as finall DIR_NAME EXTENSION\n");
      exit(1);
   }

   u = getusernames(UNAME);

   extn = (char *)malloc((strlen(argv[2]) + 2) * sizeof(char));
   sprintf(extn, ".%s", argv[2]);

   printf("NO        : OWNER               SIZE                 NAME\n");
   printf("--          -----               ----                 ----\n");
   n = findall(argv[1], extn, 0, UNAME, u);
   printf("+++ %d files match the extension %s\n", n, argv[2]);

   exit(0);
}
