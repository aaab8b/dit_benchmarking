# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 \
#                                 --mixed_precision fp16 train.py --model DiT-XL/2 \
#                                 --feature-path /root/workspace/datasets/Imagenet_DiT_feat \
#                                 --mixed-precision no \
#                                 --exp-name mi250


CUDA_VISIBLE_DEVICES=0,2 accelerate launch --main_process_port 12345 --multi_gpu --num_processes 2 \
                                --mixed_precision fp16 train.py --model DiT-XL/2 \
                                --feature-path /scratch1_nvme_1/workspace/datasets/Imagenet_DiT_feat \
                                --mixed-precision no \
                                --exp-name nv