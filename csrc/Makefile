ARCH   	= -DON_KP920
INCLUDE := -I/home/fx/lib/icl-papi/include/ -I ./

CC      := gcc
LINKER  := $(CC)
CFLAGS  := -fopenmp -fPIC -march=native -ftree-vectorize $(ARCH) $(INCLUDE) -O3
LDFLAGS := -fopenmp -fPIC -shared

Objects := meformer.o \
		gemm.o \
		fused_scalexqxkt_mask_max_kernel.o \
		fused_exp_sum_scorexv_kernel_v1.o \
		fused_exp_sum_scorexv_kernel_v2.o \
		cache_kp920.o

lib: $(Objects)
	$(LINKER) $^ -o libmeformer.so $(LDFLAGS)
	ar rcs libmeformer.a $^

bench_pre_fuse: lib
	$(CC) bench_pre_fuse.c -o bench_pre_fuse.x $(INCLUDE) $(CFLAGS) libmeformer.a \
		-I /home/fx/lib/openblas_s/include/ \
		/home/fx/lib/openblas_s/lib/libopenblas.a

bench_mid_fuse: lib
	$(CC) bench_mid_fuse.c -o bench_mid_fuse.x $(INCLUDE) $(CFLAGS) libmeformer.a \
		-I /home/fx/lib/openblas_s/include/ \
		/home/fx/lib/openblas_s/lib/libopenblas.a

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS) $(INCLUDE)

.PHONY: clean, lib

clean:
	rm *.o *.x *.a *.so