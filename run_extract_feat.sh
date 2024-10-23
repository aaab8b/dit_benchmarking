CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=8 extract_features.py --model DiT-XL/2 --data-path /root/tongs/data/ILSVRC/Data/CLS-LOC/train/ --features-path /root/workspace1/datasets/Imagenet_DiT_feat


# for ((i=0; i<=7; i++))
# do
#     CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --master_port=123$i --nproc_per_node=1 extract_features.py \
#                                 --model DiT-XL/2 --data-path /root/tongs/data/ILSVRC/Data/CLS-LOC/train/ \
#                                 --features-path /root/workspace1/datasets/Imagenet_DiT_feat \
#                                 --th_ind $i \
#                                 --num_th 8 &
# done