#!/usr/bin/env bash

dataset=imagenet
rootdir=/home/ubuntu
exp_id=resnet50_cnsn_consist
model=resnet50
wd=1e-4
epoch=90
batch_size=128
crop=both
beta=1
cnsn_type=sn
consist_wt=10
pos=post
cn_prob=0.5
gpu=0,1,2,3

CUDA_VISIBLE_DEVICES=${gpu} python imagenet.py \
    --data_dir ${rootdir}/data/imagenet/raw-data \
    --exp_dir ${rootdir}/exp \
    --corrupt_data_dir ${rootdir}/data/ImageNet-C \
    --dataset ${dataset} \
    --batch_size ${batch_size} \
    --model ${model} \
    --lr 0.1 \
    --momentum 0.9 \
    --weight_decay ${wd} \
    --workers 4 \
    --exp_id ${exp_id} \
    --epochs ${epoch} \
    --pos ${pos} \
    --cnsn_type ${cnsn_type} \
    --consist_wt ${consist_wt} \
    --cn_prob ${cn_prob} \
    --crop ${crop} \
    --beta ${beta} \
