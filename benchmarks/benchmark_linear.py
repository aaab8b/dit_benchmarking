import torch
import os



print("Tuning enabled")
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"  # Enable tuning
os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "1"  # Enable tuning
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "linear_tune_res.csv"  # Specify output file

def time_linear_forward(bs, seq_len, in_c, out_c):

    linear = torch.nn.Linear(in_c, out_c).cuda()
    in_data = torch.randn((bs, seq_len, in_c)).cuda()

    n_iter = 1000  # Number of iterations to time
    n_warmup = 10  # Number of warmup iterations

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    for i in range(n_iter + n_warmup):
        if i == n_warmup:
            t0.record()  # Don't start recording until warmup is finished
        out = linear(in_data)

    # Compute elapsed time
    t1.record()
    torch.cuda.synchronize()
    dt = t0.elapsed_time(t1) / 1000

    return n_iter/dt, dt

def time_linear_forward_backward(bs, seq_len, in_c, out_c):

    linear = torch.nn.Linear(in_c, out_c).cuda()
    in_data = torch.randn((bs, seq_len, in_c)).cuda()

    n_iter = 1000  # Number of iterations to time
    n_warmup = 10  # Number of warmup iterations

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    for i in range(n_iter + n_warmup):
        if i == n_warmup:
            t0.record()  # Don't start recording until warmup is finished
        out = linear(in_data).mean()
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
        for in_c, out_c in [(1152, 32), (1152, 6912), (4608, 1152), (1152, 4608), (1152, 1152),
                            (1152, 3456)]:
                iter_per_sec, t = time_linear_forward_backward(bs, seq_len, in_c, out_c)
                print(f'{seq_len}, {bs}, {in_c}, {out_c}, {iter_per_sec}')
                # print(f"{iter_per_sec:0.2f} iter/s ({t:0.4g}s)")
