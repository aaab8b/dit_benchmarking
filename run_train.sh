MAX_STEP=1000
RES_FOLDER='amd_res'
DATA_FOLDER='/root/workspace/datasets/Imagenet_DiT_feat'

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 \
                                --mixed_precision no train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision no \
                                --exp-name bs256_2gpu_mi250_fp32 


CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 \
                                --mixed_precision fp16 train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision fp16 \
                                --exp-name bs256_2gpu_mi250_fp16

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 \
                                --mixed_precision bf16 train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision bf16 \
                                --exp-name bs256_2gpu_mi250_bf16




# CUDA_VISIBLE_DEVICES=0,2 accelerate launch --main_process_port 12345 --multi_gpu --num_processes 2 \
#                                 --mixed_precision fp16 train.py --model DiT-XL/2 \
#                                 --feature-path /scratch1_nvme_1/workspace/datasets/Imagenet_DiT_feat \
#                                 --results-dir nv_res \
#                                 --mixed-precision no \
#                                 --exp-name nv