CC = gcc
CFLAGS = -Wall -Wextra -g

all: Q1_null Q2_leak Q3_oob Q4_use_after_free Q5_invalid_free

Q1_null: Q1_null.c
	$(CC) $(CFLAGS) -o $@ $<

Q2_leak: Q2_leak.c
	$(CC) $(CFLAGS) -o $@ $<

Q3_oob: Q3_oob.c
	$(CC) $(CFLAGS) -o $@ $<

Q4_use_after_free: Q4_use_after_free.c
	$(CC) $(CFLAGS) -o $@ $<

Q5_invalid_free: Q5_invalid_free.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f Q1_null Q2_leak Q3_oob Q4_use_after_free Q5_invalid_free
