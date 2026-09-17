# Built on the board (aarch64): make && ./bench_gemv2 ~/vectors
CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=gnu11
OBJS     = udmabuf.o gemv_dev.o gemv2_dev.o

all: bench_gemv2 libkv260gemv.a

libkv260gemv.a: $(OBJS)
	ar rcs $@ $^

# Exactness against the contract's vectors, then cold and prefetched GB/s.
# Needs the gemv2 overlay and the n32 vector cases.
bench_gemv2: bench_gemv2.o libkv260gemv.a
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c udmabuf.h gemv_dev.h gemv2_dev.h xgemv_q4_0_hw.h xgemv_fetch_hw.h xgemv_mac_hw.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.a bench_gemv2
