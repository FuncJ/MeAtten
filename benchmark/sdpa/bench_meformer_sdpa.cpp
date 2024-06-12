#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <algorithm>
#include <sys/time.h>
#include <xnnpack.h>
#include <pthreadpool.h>
#include <unistd.h>
#include "meformer.h"

#define REPEAT 3
// #define THREADS 64

void compareTensor(int n_batch, int n_head, int seq_len, int head_dim,
				std::vector<float> get, std::vector<float> expect)
{
	int batch_size, size_head;
	batch_size=(n_batch*n_head), size_head=(seq_len*head_dim);

	int flag=0;

	for(int bs=0; bs<batch_size; bs++)
	{
		for(int i=0; i<seq_len; i++)
		{
			for(int j=0; j<head_dim; j++)
			{
				float diff=(get.data()[(bs*size_head)+(i*head_dim+j)] -
						 expect.data()[(bs*size_head)+(i*head_dim+j)]);
				if((diff > 1.0e-3) || (diff < -1.0e-3))
				{
					printf("(bs:%d, i:%d, j:%d), diff=%lf, get=%lf, expect=%lf\n", bs, i, j,
													diff,
													get.data()[(bs*size_head)+(i*head_dim+j)],
												 expect.data()[(bs*size_head)+(i*head_dim+j)] );
					flag=1;
				}
				if(flag==1)
					break;
			}
			if(flag==1)
				break;
		}
		if(flag==1)
				break;
	}

	if(flag==0)
    	printf("no diff\n");
}


double test_xnnpack_sdpa(size_t n_batch, size_t n_head, size_t seq_len, size_t head_dim,
	std::vector<float> &query, std::vector<float> &key, std::vector<float> &value, std::vector<float> &scale,
	std::vector<float> &mask, std::vector<float> &output, int THREADS)
{
	const pthreadpool_t tpool=pthreadpool_create(THREADS);

	// Create, setup, run, and destroy Scaled Dot Attention operator.
	xnn_operator_t attention_op = nullptr;
	xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f32(
		xnn_attention_logits_cap_type_none,
		nullptr,
		0,
		&attention_op);

	size_t workspace_size = 0;
	size_t workspace_alignment = 0;
	status = xnn_reshape_scaled_dot_product_attention_nhtc_f32(
			attention_op,
			n_batch, n_head, seq_len, n_head, seq_len,
			head_dim, head_dim,
			&workspace_size, &workspace_alignment,
			tpool);

	std::vector<char> workspace(workspace_size, 0);
	status = xnn_setup_scaled_dot_product_attention_nhtc_f32(
			attention_op,
			workspace.data(), query.data(), key.data(), value.data(),
			scale.data(), mask.data(), output.data());

	double start=dClock(), cost;
	status=xnn_run_operator(attention_op, tpool);
	cost=(dClock()-start);

	status = xnn_delete_operator(attention_op);

	return cost;
}

double test_meformer_sdpa(size_t n_batch, size_t n_head, size_t seq_len, size_t head_dim,
	std::vector<float> &query, std::vector<float> &key, std::vector<float> &value, std::vector<float> &scale,
	std::vector<float> &mask, std::vector<float> &output, int THREADS)
{
	int num_T=THREADS;

	std::vector<float> buffer_key(num_T*head_dim*r2); /*todo*/
	std::vector<float> score(n_batch*n_head*seq_len*seq_len); /*todo*/

	std::vector<float> max_per_line(n_batch*n_head*seq_len*2);
	std::vector<float> exp_sum_per_line(n_batch*n_head*seq_len, 0);

	double start=dClock(), cost;
	scaled_dot_product_attention(n_batch, n_head, seq_len, head_dim,
			query.data(), key.data(), buffer_key.data(), score.data(),
			mask.data(),
			value.data(), output.data(), head_dim,
			max_per_line.data(), exp_sum_per_line.data());
	cost=(dClock()-start);

	return cost;
}

int main(int argc, const char *argv[])
{
	if(argc < 4) {
        std::cout << "no n_batch and n_head" << std::endl;
        return 0;
    }
	int n_batch=atoi(argv[1]), n_head=atoi(argv[2]), seq_len=atoi(argv[3]), head_dim=64;
	int THREADS=atoi(argv[4]);

	std::cout << "MEFORMER"
			<< "  THREADS: " << THREADS
			<< "  REPEAT: " << REPEAT
			<< std::endl << std::endl;

	std::vector<float> scale(XNN_EXTRA_BYTES/sizeof(float)+           head_dim);
	std::vector<float> query(XNN_EXTRA_BYTES/sizeof(float)+n_batch*n_head*seq_len*head_dim);
	std::vector<float> key  (XNN_EXTRA_BYTES/sizeof(float)+n_batch*n_head*seq_len*head_dim);
	std::vector<float> value(XNN_EXTRA_BYTES/sizeof(float)+n_batch*n_head*seq_len*head_dim);
	std::vector<float> mask (XNN_EXTRA_BYTES/sizeof(float)+seq_len*seq_len);

	std::vector<float> output    (n_batch*n_head*seq_len*head_dim);
	std::vector<float> output_ref(n_batch*n_head*seq_len*head_dim);

	std::random_device random_device;
	auto rng = std::mt19937(random_device());
	std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> scaledist(sqrt(1.0/head_dim), sqrt(1.0/head_dim));

	std::generate(scale.begin(), scale.end(), [&]() { return scaledist(rng); });
	std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
	std::generate(key.begin(), key.end(), [&]()     { return f32dist(rng); });
	std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
	std::generate(mask.begin(), mask.end(), [&]()   {return f32dist(rng); });

	// for(seq_len=80; seq_len<=1600; seq_len+=80) {

		// warm up: load code to the cache
		for(int i=0; i<1; i++) {
			test_meformer_sdpa(n_batch, n_head, seq_len, head_dim, query, key, value,
															scale, mask, output, THREADS);
		}

		// time it
		double time_cost=0;
		for(int rep=0; rep<REPEAT; rep++) {
			time_cost+=test_meformer_sdpa(n_batch, n_head, seq_len, head_dim, query, key, value,
															scale, mask, output, THREADS);
		}

		// check results
		{
            xnn_status status = xnn_initialize(nullptr);
			test_xnnpack_sdpa(n_batch, n_head, seq_len, head_dim, query, key, value,
										scale, mask, output_ref, THREADS);

			compareTensor(n_batch, n_head, seq_len, head_dim,
										output, output_ref);
            status = xnn_deinitialize();
		}

		// calculate gflops
		{
			double ops, gflops;
			ops=( n_batch*n_head*(2.0*seq_len*seq_len*head_dim + 2.0*seq_len*head_dim*seq_len) );
			gflops=( REPEAT*ops/time_cost/1.0e9 );

			FILE* fp;
			fp = fopen("run_bench_meformer_sdpa.txt", "a+");
			if(fp==NULL)
				std::cout << "fopen error" << std::endl;
			else {
				fprintf(fp, "MEFORMER THREADS:%d REPEAT:%d    batch_size:%d    heads:%d    seq_len:%d    head_dim:%d    time_cost:%.6lf     gflops:%.6lf", 
						THREADS, REPEAT,
						n_batch, n_head, seq_len, head_dim, time_cost, gflops);
				fprintf(fp, "\n");
			}

			fclose(fp);

			std::cout << "batch_size:" << n_batch
						<< "  heads:" << n_head
						<< "  seq_len:" << seq_len
						<< "  head_dim:" << head_dim
						<< "  time_cost: " << time_cost
						<< "  gflops:" << gflops 
						<< std::endl << std::endl;
		}
	// }

	return 0;
}
