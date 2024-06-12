# ----------------------------------- benchmark -----------------------------------
bench_meformer_sdpa.x: bench_meformer_sdpa.cpp
	g++ $^ -o $@ -I /home/fx/lib/xnnpack/include/ \
	    -I ../../csrc/ \
		\
		/home/fx/lib/xnnpack/lib64/libXNNPACK.a \
		/home/fx/lib/xnnpack/lib64/libcpuinfo.a \
		/home/fx/lib/xnnpack/lib64/libpthreadpool.a \
		\
		../../csrc/libmeformer.a \
		/home/fx/lib/openblas_s/lib/libopenblas.a \
		\
		-fopenmp -pthread -O3


.PHONY: clean

clean:
	rm *.x