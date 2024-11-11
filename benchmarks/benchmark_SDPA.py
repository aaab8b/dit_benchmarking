import torch
import os
import torch.nn.functional as F



print("Tuning enabled")
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"  # Enable tuning
os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "1"  # Enable tuning
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "sdpa_tune_res.csv"  # Specify output file

def time_SDPA_forward(bs, seq_len, num_heads, head_dim):

    in_data = torch.randn((bs, num_heads, seq_len, head_dim)).cuda()

    n_iter = 1000  # Number of iterations to time
    n_warmup = 10  # Number of warmup iterations

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    for i in range(n_iter + n_warmup):
        if i == n_warmup:
            t0.record()  # Don't start recording until warmup is finished
        out = F.scaled_dot_product_attention(in_data, in_data, in_data)

    # Compute elapsed time
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000

    return n_iter/dt, dt

def time_SDPA_forward_backward(bs, seq_len, num_heads, head_dim):

    in_data = torch.randn((bs, num_heads, seq_len, head_dim), requires_grad=True).cuda()

    n_iter = 1000  # Number of iterations to time
    n_warmup = 10  # Number of warmup iterations

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    for i in range(n_iter + n_warmup):
        if i == n_warmup:
            t0.record()  # Don't start recording until warmup is finished
        out = F.scaled_dot_product_attention(in_data, in_data, in_data).mean()
        out.backward()
        

    # Compute elapsed time
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000

    return n_iter/dt, dt

    # print(f"{n_iter/dt:0.2f} iter/s ({dt:0.4g}s)")
print(f'seq_len, bs, in_c, out_c, iter_per_sec')
for seq_len in [256, 1024]:
    for bs in [16, 32, 64]:
        for num_heads, head_dim in [(16, 72)]:
                iter_per_sec, t = time_SDPA_forward(bs, seq_len, num_heads, head_dim)
                print(f'{seq_len}, {bs}, {num_heads}, {head_dim}, {iter_per_sec}')
                # print(f"{iter_per_sec:0.2f} iter/s ({t:0.4g}s)")
