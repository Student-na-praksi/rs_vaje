GEM5_PATH = /d/hpc/projects/FRI/GEM5/gem5_workspace/gem5/
ISA = riscv
GXX = riscv64-linux-gnu-gcc

# Source files
SOURCES = $(wildcard *.c)
BINARIES = $(SOURCES:.c=.bin)

all: $(BINARIES)

%.bin: %.c
	$(GXX) -o $@ $< -static -fopenmp -lpthread -I$(GEM5_PATH)/include -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -I$(GEM5_PATH)/util/m5/src -march=rv64gcv -lm5 -DGEM5

clean:
	rm -f *.bin