NVCC=nvcc

CUDAFLAGS= -arch=sm_70

OPT= -g -G

RM=/bin/rm -f

all: test


main: test.o gqf.o

        ${NVCC} ${OPT} -o main test.o gqf.o


gqf.o: gqf.cuh gqf.cu

        ${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c Generate.cpp


IC.o: Header.cuh IC.cu

        $(NVCC) ${OPT} $(CUDAFLAGS)        -std=c++11 -c IC.cu


IC: IC.o Generate.o

        ${NVCC} ${CUDAFLAGS} -o IC IC.o Generate.o

clean:

        ${RM} *.o IC