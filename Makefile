CC=gcc
CFLAGS=-g -O3
LFLAGS=

all: testMatrix

testMatrix: testMatrix.o fmtx.o libfixmath/libfixmath/fix16.o
        $(CC) $(CFLAGS) $^ -o $@

%.o : $.c
        $(CC) $(CFLAGS) -c $<


clean: 
        rm -rf *.o testMatrix *~