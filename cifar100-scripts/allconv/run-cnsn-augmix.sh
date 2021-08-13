#!/usr/bin/env bash

dataset=CIFAR-100
rootdir=/home/ubuntu
exp_id=allconv_cnsn_augmix
model=allconv
wd=5e-4
epoch=100
batch_size=128
crop=style
beta=1
cnsn_type=cnsn
consist_wt=10
pos=1
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
    --lr 0.1 \
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
