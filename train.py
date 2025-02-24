# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator
from tqdm import tqdm

from models import DiT_models
from diffusion import create_diffusion,FlashAttnProcessor2_0,set_attn_processor
from accelerate.utils import set_seed
from diffusers import DiTTransformer2DModel
import wandb


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # name = name.replace("_orig_mod._orig_mod.","_orig_mod.")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if name not in ema_params.keys():
            print(name)
            print("ema")
            print(ema_params.keys())
            print("model")
            print(model_params.keys())
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))

        self.retries = 20

    def __len__(self):
        return len(self.features_files)

    def __getitem__(self, idx):
        cur_retry = 0
        while cur_retry < self.retries: 
            try:
                feature_file = self.features_files[idx]

                features = np.load(os.path.join(self.features_dir, feature_file))
                labels = int(feature_file[feature_file.rfind('label')+5:feature_file.rfind('.npy')])
            except Exception as e:
                print('error when loading file: %s' % feature_file)
                print(e)
                print('retrying...')
                if cur_retry < self.retries:
                    cur_retry += 1
                    idx = np.random.randint(0, len(self.features_files)-1)
                    continue
            break
        return torch.from_numpy(features), torch.tensor([labels])


class DummyDataset(Dataset):
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        feat = torch.zeros((4, self.latent_dim, self.latent_dim))
        label = torch.zeros((1)).long()
        return feat, label

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with='wandb',
    )
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        accelerator.init_trackers(
            project_name="DiT", 
            config=args,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
    if args.gemm_tuning:
        if accelerator.is_main_process:
            logger.info("using gemm tuning")
        #use TUNABLEOP and gemm tuning csv
        os.environ["PYTORCH_TUNABLEOP_VERBOSE"]="1"
        os.environ["PYTORCH_TUNABLEOP_ENABLED"]="1"
        os.environ["PYTORCH_TUNABLEOP_FILENAME"]="gemm_tuning_results/dit_{}.csv".format(args.image_size)
    if args.seed is not None:
        set_seed(args.seed)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    # model = DiT_models[args.model](
    #     input_size=latent_size,
    #     num_classes=args.num_classes,
    #     use_fa=args.use_fa
    # )
    #I checked default settings of DiTTransformer2DModel, which is XL/2.
    model=DiTTransformer2DModel(
        sample_size=latent_size,
        num_embeds_ada_norm=args.num_classes,
        in_channels=4,
        out_channels=8
        )
    if args.use_fa:
        if accelerator.is_main_process:
            logger.info("using flash attention")
        set_attn_processor(model,FlashAttnProcessor2_0())
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    if args.compile:
        if accelerator.is_main_process:
            logger.info("using torch compile")
        torch._dynamo.config.optimize_ddp=False
        model= torch.compile(model)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    if args.dummydata:
        dataset = DummyDataset(latent_size)
    else:
        dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    # running_loss = 0
    # start_time = time()

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=train_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    WARMUP_ITERS = 90

    iter_count = 0
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            # print("x:{}".format(x.shape))
            # print("y:{}".format(y.shape))
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # model_kwargs = dict(y=y)
            model_kwargs= dict(class_labels=y,return_dict=False)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            train_steps += 1
            progress_bar.update(1)
            # print(accelerator.gather(loss))
            logs = {
                "loss": accelerator.gather(loss).mean().detach().item(),
            }
            accelerator.log(logs, step=train_steps)
            progress_bar.set_postfix(**logs)

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            if train_steps == WARMUP_ITERS:
                t0.record()

            if train_steps >= args.max_train_steps:
                break
        if train_steps >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    if accelerator.is_main_process:
        logger.info("Done!")
        
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1) / 1000

        with open("benchmarking_result.txt",mode="a") as f:
            f.write(f"image_size:{args.image_size},world_size:{accelerator.num_processes},batch_size:{args.global_batch_size},{(train_steps-WARMUP_ITERS)*args.global_batch_size/dt:0.2f} samples/s ({dt:0.4g}s)\n")
        logger.info(f"{(train_steps-WARMUP_ITERS)*args.global_batch_size/dt:0.2f} samples/s ({dt:0.4g}s)")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--max-train-steps", type=int, default=400_000)
    parser.add_argument(
        "--dummydata",
        action='store_true',
        help="whether to use dummy data",
    )
    parser.add_argument("--exp-name", type=str, default="init_exp")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument(
        "--use_fa",
        action='store_true',
        help="whether to use flash attention",
    )
    parser.add_argument(
        "--compile",
        action='store_true',
        help="whether to use torch.compile",
    )
    parser.add_argument(
        "--gemm-tuning",
        action='store_true',
        help="whether to use torch.compile",
    )

    args = parser.parse_args()
    main(args)
