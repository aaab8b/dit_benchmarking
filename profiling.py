from models import DiT_models
from torch.utils.data import Dataset, DataLoader
from train import CustomDataset
from diffusion import create_diffusion
import torch
from tqdm import tqdm
import os
from torch.profiler import profile, record_function, ProfilerActivity

DATA_ROOT = '../../datasets/Imagenet_DiT_feat_sub/imagenet256_features'
BS = 64
WARMUP_ITERS = 10
MODEL = 'DiT-XL/2'
DEVICE = 'cuda'
TUNE = False
PROFILING = True


activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

if TUNE:
    print("Tuning enabled")
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"  # Enable tuning
    os.environ["PYTORCH_TUNABLEOP_VERBOSE"] = "1"  # Enable tuning
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "tune_res.csv"  # Specify output file



dataset = CustomDataset(DATA_ROOT, '')
loader = DataLoader(
    dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)


t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)

model = DiT_models[MODEL](
        input_size=32,
        num_classes=1000
    ).to(DEVICE)

diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule


iter_count = 0
for x, y in tqdm(loader):
        
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    x = x.squeeze(dim=1)
    y = y.squeeze(dim=1)
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=DEVICE)
    model_kwargs = dict(y=y)
    with torch.no_grad():
        if PROFILING:
            with profile(activities=activities) as prof:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            prof.export_chrome_trace("trace.json")
            exit()
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
    iter_count += 1
    if iter_count == WARMUP_ITERS:
        t0.record()

t1.record()
torch.cuda.synchronize()
dt = t0.elapsed_time(t1) / 1000


print(f"{(iter_count-WARMUP_ITERS)*BS/dt:0.2f} samples/s ({dt:0.4g}s)")