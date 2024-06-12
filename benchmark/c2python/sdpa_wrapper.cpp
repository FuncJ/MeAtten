#include <torch/extension.h>
#include <xnnpack.h>
#include <pthreadpool.h>
#include <vector>
#include <cmath>

void *buffer, *out;
int buffer_size=0, out_size=0;

void fast_malloc(int buffer_size_0, int out_size_0)
{
	if((buffer_size_0+out_size_0) != (buffer_size+out_size)) {

		buffer_size=buffer_size_0, out_size=out_size_0;

		if(buffer!=NULL)
			free(buffer);
		posix_memalign(&buffer, 1024, buffer_size*sizeof(float));
		if(out!=NULL)
			free(out);
		posix_memalign(&out, 1024, out_size*sizeof(float));
	}
}

extern "C" {
	void scaled_dot_product_attention(long n_batch, long n_head, long seq_len, long head_dim,
						float *q_data, float *kt_data, float *buffer_kt_out, float *score_data_out,
						float *mask_data,
						float *v_data, float *out_data, long ldo,
						float *max_per_line_out,
						float *exp_sum_per_line_out);
}

torch::Tensor meatten_sdpa(
	int64_t n_batch, int64_t n_head,  int64_t seq_len, int64_t head_dim,
	torch::Tensor &q_data, torch::Tensor &kt_data,
	torch::Tensor &mask_data,
	torch::Tensor &v_data
)
{
	// std::cout << "meatten" << std::endl;

	int buffer_key_size=(64*head_dim*16);
	int buffer_score_size=(64*320*320);
	int buffer_max_size=(n_batch*n_head*seq_len*2);
	int buffer_exp_size=(n_batch*n_head*seq_len);
	int buffer_size_0=(buffer_key_size+buffer_score_size+buffer_max_size+buffer_exp_size);
	// float *buffer=(float *)malloc(sizeof(float)*buffer_size);

	int out_size_0=(n_batch*n_head*seq_len*head_dim);
	// float *out=(float *)malloc(sizeof(float)*out_size);

	fast_malloc(buffer_size_0, out_size_0);

	scaled_dot_product_attention( n_batch, n_head, seq_len, head_dim,
		(float *)q_data.data_ptr(), (float *)kt_data.data_ptr(), (float *)buffer, (float *)(buffer)+buffer_key_size,
		(float *)mask_data.data_ptr(),
		(float *)v_data.data_ptr(), (float *)out, head_dim,
		(float *)(buffer)+buffer_key_size+buffer_score_size,
		(float *)(buffer)+buffer_key_size+buffer_score_size+buffer_max_size );

	torch::Tensor res = torch::from_blob(out, {n_batch, n_head, seq_len, head_dim});
	return res;
}

void xnn_sdpa(
	int64_t n_batch, int64_t n_head,  int64_t seq_len, int64_t head_dim,
	torch::Tensor &query, torch::Tensor &key,
	torch::Tensor &mask,
	torch::Tensor &value, torch::Tensor &out
)
{
	// init
	xnn_status status = xnn_initialize(nullptr);
	const pthreadpool_t tpool=pthreadpool_create(64);
	std::vector<float> scale(head_dim, 1/sqrt(head_dim));

	// Create, setup, run, and destroy Scaled Dot Attention operator.
	xnn_operator_t attention_op = nullptr;
	status = xnn_create_scaled_dot_product_attention_nhtc_f32(
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
			workspace.data(), (float *)query.data_ptr(), (float *)key.data_ptr(), (float *)value.data_ptr(),
			scale.data(), (float *)mask.data_ptr(), (float *)out.data_ptr());

	status=xnn_run_operator(attention_op, tpool);

	// deinit
	status = xnn_delete_operator(attention_op);
	status = xnn_deinitialize();




	// xnn_status status = xnn_initialize(nullptr);
	// const pthreadpool_t tpool=pthreadpool_create(64);

	// // ------------------------------------------------------------------------------------------
	// xnn_operator_t q_scale_mul_op = nullptr;
	// status = xnn_create_multiply_nd_f32(-std::numeric_limits<float>::infinity(),
	// 	std::numeric_limits<float>::infinity(), 0, &q_scale_mul_op);

	// xnn_operator_t qk_bmm_op = nullptr;
	// status = xnn_create_batch_matrix_multiply_nc_f32(XNN_FLAG_TRANSPOSE_B, &qk_bmm_op);

	// xnn_operator_t add_op = nullptr;
	// status = xnn_create_add_nd_f32(-std::numeric_limits<float>::infinity(),
	// 	std::numeric_limits<float>::infinity(), 0, &add_op);

	// xnn_operator_t softmax_op = nullptr;
	// status = xnn_create_softmax_nc_f32(seq_len, seq_len, seq_len, 0, &softmax_op);

	// xnn_operator_t attn_value_bmm_op = nullptr;
	// status = xnn_create_batch_matrix_multiply_nc_f32(0, &attn_value_bmm_op);

	// // ------------------------------------------------------------------------------------------
	// std::array<size_t, 4> query_dims = {n_batch, n_head, seq_len, head_dim};
	// std::array<size_t, 1> scale_dims = {head_dim};
	// status = xnn_reshape_multiply_nd_f32(q_scale_mul_op,
	// 	query_dims.size(), query_dims.data(),
	// 	scale_dims.size(), scale_dims.data(), tpool);

	// size_t workspace_size = 0;
	// size_t workspace_alignment = 0;
	// status = xnn_reshape_batch_matrix_multiply_nc_f32(qk_bmm_op,
	// 		n_batch*n_head, seq_len, head_dim, seq_len,
	// 		&workspace_size, &workspace_alignment, tpool);

	// std::array<size_t, 4> logits_dims = {n_batch, n_head, seq_len, seq_len};
	// std::array<size_t, 2> mask_dims = {seq_len, seq_len};
	// status = xnn_reshape_add_nd_f32(add_op,
	// 		logits_dims.size(), logits_dims.data(),
	// 		mask_dims.size(), mask_dims.data(), tpool);

	// status = xnn_reshape_softmax_nc_f32(softmax_op, n_batch*n_head*seq_len, tpool);

	// size_t workspace_size2 = 0;
	// size_t workspace_alignment2 = 0;
	// status = xnn_reshape_batch_matrix_multiply_nc_f32(attn_value_bmm_op,
	// 	n_batch*n_head, seq_len, seq_len, head_dim,
	// 	&workspace_size2, &workspace_alignment2, tpool);

	// // ------------------------------------------------------------------------------------------
	// std::vector<float> scale(head_dim, 1/sqrt(head_dim));
	// std::vector<float> query_scaled(XNN_EXTRA_BYTES/sizeof(float) + n_batch*n_head*seq_len*head_dim);
	// status = xnn_setup_multiply_nd_f32(q_scale_mul_op,
	// 	(float *)query.data_ptr(), scale.data(), query_scaled.data());

	// std::vector<float> logits(XNN_EXTRA_BYTES/sizeof(float) + n_batch*n_head*seq_len*seq_len);
	// std::vector<char> workspace(workspace_size, 0);
	// status = xnn_setup_batch_matrix_multiply_nc_f32(qk_bmm_op,
	// 	workspace.data(), query_scaled.data(), (float *)key.data_ptr(), logits.data());
	
	// status = xnn_setup_add_nd_f32(add_op,
	// 			logits.data(), (float *)mask.data_ptr(), logits.data());

	// status = xnn_setup_softmax_nc_f32(softmax_op, logits.data(), logits.data());
	
	// std::vector<char> workspace2(workspace_size2, 0);
	// status = xnn_setup_batch_matrix_multiply_nc_f32(attn_value_bmm_op,
	// 	workspace2.data(), logits.data(), (float *)value.data_ptr(), (float *)out.data_ptr());

	// // ------------------------------------------------------------------------------------------
	// xnn_run_operator(q_scale_mul_op, tpool);
	// xnn_run_operator(qk_bmm_op, tpool);

	// xnn_run_operator(add_op, tpool);
	// for(int i=0; i<seq_len; i++) {
	// 	for(int j=0; j<seq_len; j++) {
	// 		std::cout << logits.data()[i*seq_len+j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;

	// xnn_run_operator(softmax_op, tpool);
	// for(int i=0; i<seq_len; i++) {
	// 	for(int j=0; j<seq_len; j++) {
	// 		std::cout << logits.data()[i*seq_len+j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << std::endl;
	
	// xnn_run_operator(attn_value_bmm_op, tpool);

	// status = xnn_delete_operator(q_scale_mul_op);
	// status = xnn_delete_operator(qk_bmm_op);
	// status = xnn_delete_operator(add_op);
	// status = xnn_delete_operator(softmax_op);
	// status = xnn_delete_operator(attn_value_bmm_op);

	// status = xnn_deinitialize();
}

void meatten_verify(
	int64_t n_batch, int64_t n_head,  int64_t seq_len, int64_t head_dim,
	torch::Tensor &get, torch::Tensor &expect
)
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
				float diff=( ( (float *)get.data_ptr() )[(bs*size_head)+(i*head_dim+j)] -
							( (float *)expect.data_ptr() )[(bs*size_head)+(i*head_dim+j)] );
				if((diff > 1.0e-3) || (diff < -1.0e-3))
				{
					printf( "(bs:%d, i:%d, j:%d), diff=%lf, get=%lf, expect=%lf\n", bs, i, j,
													diff,
													( (float *)get.data_ptr() )[(bs*size_head)+(i*head_dim+j)],
													( (float *)expect.data_ptr() )[(bs*size_head)+(i*head_dim+j)] );
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

TORCH_LIBRARY(clib, m) {
	m.def("meatten_sdpa", meatten_sdpa);
	m.def("xnn_sdpa", xnn_sdpa);
	m.def("meatten_verify", meatten_verify);
}
