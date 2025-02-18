import pandas as pd
import os
os.environ["PYTORCH_TUNABLEOP_VERBOSE"]="1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"]="1"
os.environ["PYTORCH_TUNABLEOP_FILENAME"]="gemm_tuning_results/profiling_dit_{}_pytorch.csv".format(256)
# info={'name': 'aten::addmm', 'shape': {'src': ((4608,), (8192, 1152), (1152, 4608), (), (), (8192, 4608)), 'MNK': (8192, 4608, 1152)}, 'dtype': ('c10::BFloat16', 'c10::BFloat16', 'c10::BFloat16', 'Scalar', 'Scalar', 'c10::BFloat16'), 'stride_info': ((1,), (1152, 1), (1, 1152), (), (), (4608, 1))}

kernel_excel_path="./MI308_model_kernel_ratio.xlsx"
df = pd.read_excel(kernel_excel_path)
import torch
import time
# from torch.profiler import profile, record_function, ProfilerActivity,schedule
torch_sum_time=[]
time_ratios=[]
# activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
# my_schedule = schedule(
# skip_first=10,
# wait=5,
# warmup=1,
# active=3,
# repeat=2)
# def trace_handler(p):
#     sort_by_keyword = "cuda_time_total"
#     # output = p.key_averages(group_by_input_shape=True).table(sort_by=sort_by_keyword, row_limit=20)
#     # print(output)
# with profile(activities=activities, record_shapes=True,schedule=my_schedule,on_trace_ready=trace_handler,with_flops=True) as prof:
for index, row in df.iterrows():
    m = row['M']
    n = row['N']
    k = row['K']
    kernel_name = row['kernel_name']
    kernel_num = row['kernel_num']
    time_sum=row["time_sum"]
    print(f"Processing row {index}: M={m}, N={n}, K={k}")
    if kernel_name == 'aten::addmm':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16
        
        # 检查设备是否支持bfloat16
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
            device = 'cpu'
        if device == 'cuda' and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Device does not support bfloat16")
        
        # 生成输入张量
        input_tensor = torch.randn(n, device=device, dtype=dtype)  # (N,)
        mat1 = torch.randn(m, k, device=device, dtype=dtype)      # (M, K)
        mat2 = torch.randn(k, n, device=device, dtype=dtype)      # (K, N)
        
        compiled_function=torch.compile(torch.addmm)

        # 预热
        with torch.no_grad():
            for _ in range(100):
                _ = compiled_function(input_tensor, mat1, mat2)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # 计时
        times = []
        with torch.no_grad():
            for _ in range(kernel_num):
                if device == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                
                _ = compiled_function(input_tensor, mat1, mat2)
                # prof.step()
                
                if device == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = start_event.elapsed_time(end_event)
                else:
                    elapsed = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
                times.append(elapsed)
        
        total_time = sum(times)
        max_time = max(times)
        min_time = min(times)
        print(f"Replicated {kernel_num} times - Total: {total_time:.2f}ms, "
            f"Max: {max_time:.2f}ms, Min: {min_time:.2f}ms\n")

    elif kernel_name == 'aten::mm':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16
        
        # 检查设备是否支持bfloat16
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead.")
            device = 'cpu'
        if device == 'cuda' and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Device does not support bfloat16")
        
        # 生成输入张量
        # input_tensor = torch.randn(n, device=device, dtype=dtype)  # (N,)
        mat1 = torch.randn(m, k, device=device, dtype=dtype)      # (M, K)
        mat2 = torch.randn(k, n, device=device, dtype=dtype)      # (K, N)
        compiled_function=torch.compile(torch.mm)
        # 预热
        with torch.no_grad():
            for _ in range(100):
                _ = compiled_function(mat1, mat2)
            if device == 'cuda':
                torch.cuda.synchronize()
        
        # 计时
        times = []
        with torch.no_grad():
            for _ in range(kernel_num):
                if device == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                
                _ = compiled_function( mat1, mat2)
                
                if device == 'cuda':
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed = start_event.elapsed_time(end_event)
                else:
                    elapsed = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
                times.append(elapsed)
        
        total_time = sum(times)
        max_time = max(times)
        min_time = min(times)
        print(f"Replicated {kernel_num} times - Total: {total_time:.2f}ms, "
            f"Max: {max_time:.2f}ms, Min: {min_time:.2f}ms\n")
    torch_sum_time.append(total_time)
    time_ratios.append(time_sum/total_time)
        # sort_by_keyword = "cuda_time_total"
        # output = prof.key_averages(group_by_input_shape=True,).table(sort_by=sort_by_keyword, row_limit=100,max_src_column_width=100,max_shapes_column_width=100,max_name_column_width=100)
        # print(output)
        # prof.export_chrome_trace("trace_use_hipblast_gemm_tuning.json")
        # torch.save(output,f"profiling_MI300_torchmm_compile.txt")
df["torch_sum_time"]=torch_sum_time
df["time_ratio_to_torch"]=time_ratios
out_file="torch_ratio_result.xlsx"
df.to_excel(out_file, sheet_name='Sheet1', index=False)