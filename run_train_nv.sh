MAX_STEP=1000
RES_FOLDER='nv_res'
DATA_FOLDER='/scratch1_nvme_1/workspace/datasets/Imagenet_DiT_feat'
gpus='0,2'

CUDA_VISIBLE_DEVICES=${gpus} accelerate launch --multi_gpu --num_processes 2 \
                                --mixed_precision no train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision no \
                                --exp-name bs256_2gpu_a100_fp32 


CUDA_VISIBLE_DEVICES=${gpus} accelerate launch --multi_gpu --num_processes 2 \
                                --mixed_precision fp16 train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision fp16 \
                                --exp-name bs256_2gpu_a100_fp16


CUDA_VISIBLE_DEVICES=${gpus} accelerate launch --multi_gpu --num_processes 2 \
                                --mixed_precision bf16 train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision bf16 \
                                --exp-name bs256_2gpu_a100_bf16



