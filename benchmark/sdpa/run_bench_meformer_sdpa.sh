# make clean
# make bench_meformer_sdpa.x


# 64 core
export OMP_NUM_THREADS=64 GOMP_CPU_AFFINITY="0-63:1"

for batch_size in 16 32 64
do
    for h in 12
    do
        for seqlen in {80..1600..80}
        do
            numactl -m 0-1 ./bench_meformer_sdpa.x $batch_size $h $seqlen 64
        done
    done
done


# 32 core
export OMP_NUM_THREADS=32 GOMP_CPU_AFFINITY="0-31:1"

for batch_size in 16 32 64
do
    for h in 12
    do
        for seqlen in {80..1600..80}
        do
            numactl -m 0 ./bench_meformer_sdpa.x $batch_size $h $seqlen 32
        done
    done
done


# scalability
OMP_NUM_THREADS=1 GOMP_CPU_AFFINITY="0" numactl -m 0 ./bench_meformer_sdpa.x 32 12 480 1
OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY="0-3:1" numactl -m 0 ./bench_meformer_sdpa.x 32 12 480 4
OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0-7:1" numactl -m 0 ./bench_meformer_sdpa.x 32 12 480 8
OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15:1" numactl -m 0 ./bench_meformer_sdpa.x 32 12 480 16
OMP_NUM_THREADS=24 GOMP_CPU_AFFINITY="0-23:1" numactl -m 0 ./bench_meformer_sdpa.x 32 12 480 24
OMP_NUM_THREADS=32 GOMP_CPU_AFFINITY="0-31:1" numactl -m 0 ./bench_meformer_sdpa.x 32 12 480 32
OMP_NUM_THREADS=40 GOMP_CPU_AFFINITY="0-39:1" numactl -m 0-1 ./bench_meformer_sdpa.x 32 12 480 40
OMP_NUM_THREADS=48 GOMP_CPU_AFFINITY="0-47:1" numactl -m 0-1 ./bench_meformer_sdpa.x 32 12 480 48
OMP_NUM_THREADS=56 GOMP_CPU_AFFINITY="0-55:1" numactl -m 0-1 ./bench_meformer_sdpa.x 32 12 480 56
OMP_NUM_THREADS=64 GOMP_CPU_AFFINITY="0-63:1" numactl -m 0-1 ./bench_meformer_sdpa.x 32 12 480 64