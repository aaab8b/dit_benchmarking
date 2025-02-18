MAX_STEP=150
RES_FOLDER='amd_res'
DATA_FOLDER='/root/workspace/datasets/Imagenet_DiT_feat'

# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 \
#                                 --mixed_precision no train.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision no \
#                                 --exp-name bs256_2gpu_mi250_fp32 

# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision fp16 train.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision fp16 \
#                                 --dummydata \
#                                 --image-size 256 \
#                                 --global-batch-size 256\
#                                 --exp-name bs256_8gpu_mi250_fp16 \
#                                 --use_fa
# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision bf16 train.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision bf16 \
#                                 --dummydata \
#                                 --image-size 256 \
#                                 --global-batch-size 256\
#                                 --exp-name bs256_8gpu_mi308_bf16_baseline

# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision bf16 train.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision bf16 \
#                                 --dummydata \
#                                 --image-size 256 \
#                                 --global-batch-size 256\
#                                 --exp-name bs256_8gpu_mi308_bf16_with_fa \
#                                 --use_fa

accelerate launch --multi_gpu --num_processes 8 \
                                --mixed_precision bf16 train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision bf16 \
                                --dummydata \
                                --image-size 256 \
                                --global-batch-size 256\
                                --exp-name bs256_8gpu_mi308_bf16_baseline_and_fa_compile_gemm_tuning \
                                --use_fa \
                                --compile \
                                --gemm-tuning
                                
                                
DISABLE_ADDMM_HIP_LT=0 TORCH_BLAS_PREFER_HIPBLASLT=1 ROCBLAS_USE_HIPBLASLT=1 accelerate launch --multi_gpu --num_processes 8 \
                                --mixed_precision bf16 train.py --model DiT-XL/2 \
                                --feature-path ${DATA_FOLDER} \
                                --max-train-steps ${MAX_STEP} \
                                --results-dir ${RES_FOLDER} \
                                --mixed-precision bf16 \
                                --dummydata \
                                --image-size 256 \
                                --global-batch-size 256\
                                --exp-name bs256_8gpu_mi308_bf16_with_fa_gemm_tuning_use_hipblaslt_only \
                                --use_fa \
                                --gemm-tuning \
                                --compile

# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision bf16 train_profiling.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision bf16 \
#                                 --dummydata \
#                                 --image-size 256 \
#                                 --global-batch-size 256\
#                                 --exp-name bs256_8gpu_mi308_bf16_with_fa_profiling_gemm_tuning \
#                                 --use_fa \
#                                 --gemm_tuning

# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision bf16 train_profiling.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision bf16 \
#                                 --dummydata \
#                                 --image-size 256 \
#                                 --global-batch-size 256\
#                                 --exp-name bs256_8gpu_mi308_bf16_with_fa_profiling \
#                                 --use_fa\
#                                 --compile

# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision bf16 train_profiling.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision bf16 \
#                                 --dummydata \
#                                 --image-size 256 \
#                                 --global-batch-size 256\
#                                 --exp-name bs256_8gpu_mi308_bf16_with_fa_profiling \
#                                 --use_fa \
#                                 --compile
# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision fp16 train.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision fp16 \
#                                 --dummydata \
#                                 --image-size 512 \
#                                 --exp-name bs512_8gpu_mi250_fp16

# accelerate launch --multi_gpu --num_processes 8 \
#                                 --mixed_precision bf16 train.py --model DiT-XL/2 \
#                                 --feature-path ${DATA_FOLDER} \
#                                 --max-train-steps ${MAX_STEP} \
#                                 --results-dir ${RES_FOLDER} \
#                                 --mixed-precision bf16 \
#                                 --dummydata \
#                                 --image-size 512 \
#                                 --exp-name bs512_8gpu_mi250_bf16



