#include "types.h"
#include "stat.h"
#include "user.h"

int
main(int argc, char *argv[])
{
  printf(1, "Enabling two-strike mode. Spinning... try killing me with Ctrl+C\n");

  twostrike(1);  // BLANK 6

  while(1) {     // BLANK 7
    // Busy loop
  }

  exit();
}
