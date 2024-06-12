import torch
import torch.nn.functional as F
import math
import numpy as np
from timeit import default_timer as timer

def verify_tensor(get, expected):
	n_batch, n_head, seq_len, head_dim=get.size(0), get.size(1), get.size(2), get.size(3)

	torch.ops.clib.meatten_verify(n_batch, n_head, seq_len, head_dim, get, expected)

def custom_func(q, k, v, mask):
	scale = math.sqrt(q.size(-1))

	score = torch.matmul(q, k.transpose(-2, -1)/scale)
	score = score + mask
	attn = F.softmax(score, dim=-1)
	o = torch.matmul(attn, v)

	return o

def pytorch_func(q, k, v, mask):
	o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

	return o

def xnn_func(n_batch, n_head, seq_len, head_dim,
			q, k,
			mask,
			v, out):
	torch.ops.clib.xnn_sdpa(
		n_batch, n_head, seq_len, head_dim,
		q, k,
		mask,
		v, out)

	return out

def meatten_func(n_batch, n_head, seq_len, head_dim,
				q, k,
				mask,
				v):
	out=torch.ops.clib.meatten_sdpa(
		n_batch, n_head, seq_len, head_dim,
		q, k,
		mask,
		v)

	return out

def test(func_name, q, k, v, mask):
	if func_name in ["custom_func", "pytorch_func"]:
		st = timer()
		o = globals()[func_name](q, k, v, mask)
		tt = timer() - st

		return o, tt
	else:
		n_batch, n_head, seq_len, head_dim = q.size(0), q.size(1), q.size(2), q.size(3)
		o = torch.rand_like(q)

		if func_name in ["xnn_func"]:
			st = timer()
			globals()[func_name](n_batch, n_head, seq_len, head_dim,
										q, k,
										mask,
										v, o)
			tt = timer() - st
			return o, tt
		else:
			st = timer()
			o = globals()[func_name](n_batch, n_head, seq_len, head_dim,
										q, k,
										mask,
										v)
			tt = timer() - st
			return o, tt

if __name__ == "__main__":
	torch.ops.load_library("build/libclib.so")

	n_batch = [32, 64]
	seq_len = [sql for sql in range(160, 1600+160, 160)]
	n_head, head_dim = 12, 64
	test_num = 3

	for bs in n_batch:
		for sql in seq_len:
			print(f"Shape n_batch:{bs}, n_head:{n_head}, seq_len:{sql}, head_dim:{head_dim}")

			q = torch.rand((bs, n_head, sql, head_dim))*2-1
			k = torch.rand((bs, n_head, sql, head_dim))*2-1
			v = torch.rand((bs, n_head, sql, head_dim))*2-1
			mask = torch.rand((sql, sql))*2-1

			# q = torch.tensor( [[[[-0.9841, -0.9506], [-0.4117, -0.8342], [-0.1665,  0.9832]]]] )
			# k = torch.tensor( [[[[0.2694,  0.4826], [0.8201, -0.7271], [0.0162,  0.9675]]]] )
			# v = torch.tensor( [[[[-0.8626,  0.4991], [-0.1451,  0.7910], [-0.9770,  0.3455]]]] )
			# mask = torch.tensor( [[0.8577 , 0.9269, 0.5293], [0.9391 , 0.6939, 0.2390], [-0.2722, 0.0062, 0.9128]] )

			# print("q\n", q)
			# print("k\n", k)
			# print("v\n", v)
			# print("mask\n", mask)

			for idx in range(test_num):
				# c_o, c_t = test("custom_func", q, k, v, mask)
				# print(f"custom func time: {c_t:.6f}")
				# print(c_o, "\n")

				pt_o, pt_t = test("pytorch_func", q, k, v, mask)
				print(f"pytorch func time: {pt_t:.6f}")
				# print(pt_o, "\n")

				# xn_o, xn_t = test("xnn_func", q, k, v, mask)
				# print(f"xnnpack func time: {xn_t:.9f}")
				# print(xn_o, "\n")

				ma_o, ma_t = test("meatten_func", q, k, v, mask)
				print(f"meatten func time: {ma_t:.9f}")
				# print(ma_o, "\n")

				ops=bs*n_head*(4.0*sql*sql*head_dim)
				gflops=ops/ma_t/1.0e9
				print("ops:%d time:%.6f gflops:%.6f" % (ops, ma_t, gflops))

				# np.testing.assert_allclose(ma_o, pt_o, rtol=1.0e-3)
				verify_tensor(ma_o, pt_o)
			print("\n")


# ### Bench Matmul
# print(torch.get_num_threads())
# m = n = k = 4096
# a = torch.rand((m, k))
# b = torch.rand((k, n))
# c = torch.rand((m, n))
# c_bak = torch.clone(c)

# c = torch.matmul(a, b)
# c_bak = torch.matmul(a, b)

# np.testing.assert_allclose(c, c_bak, rtol=1.0e-3)

# ss = timer()
# for i in range(5):
# 	c = torch.matmul(a, b)
# tt = timer() - ss

# ops=5*2.0*m*n*k
# gflops=ops/tt/1.0e9

# print(gflops)