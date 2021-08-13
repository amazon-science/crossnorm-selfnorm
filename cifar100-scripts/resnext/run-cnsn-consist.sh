#!/usr/bin/env bash

dataset=CIFAR-100
rootdir=/home/ubuntu
exp_id=resnext_cnsn_consist
model=resnext
wd=5e-4
epoch=200
lr=0.05 # remember to change it back to 0.1 if setting batch_size to 128
batch_size=64 # this was only to fit 1 K80 GPU
crop=neither
beta=1
cnsn_type=cnsn
consist_wt=10
pos=post
cn_prob=0.25
active_num=1
gpu=3

CUDA_VISIBLE_DEVICES=${gpu} python cifar.py \
    --data_dir ${rootdir}/data \
    --exp_dir ${rootdir}/exp \
    --corrupt_data_dir ${rootdir}/data/${dataset}-C \
    --dataset ${dataset} \
    --batch_size ${batch_size} \
    --model ${model} \
    --lr ${lr} \
    --momentum 0.9 \
    --weight_decay ${wd} \
    --workers 2 \
    --exp_id ${exp_id} \
    --epochs ${epoch} \
    --pos ${pos} \
    --cnsn_type ${cnsn_type} \
    --consist_wt ${consist_wt} \
    --cn_prob ${cn_prob} \
    --active_num ${active_num} \
    --crop ${crop} \
    --beta ${beta} \
