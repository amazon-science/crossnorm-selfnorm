#!/usr/bin/env bash

dataset=CIFAR-10
rootdir=/home/ubuntu
exp_id=resnext_sn
model=resnext
wd=5e-4
epoch=200
batch_size=128
cnsn_type=sn
pos=pre
gpu=1

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
