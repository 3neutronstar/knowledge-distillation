#!/bin/bash
MODE="train_moo  baseline_moo";  
SET="0.25 0.375 0.5 0.625 0.75";
SPARSE_SET="";
for VALUE in ${SET}
do
    for SPARSE_VALUE in ${SPARSE_SET}
    do
        for MODE_VAL in ${MODE}
        do
            python run.py train_selfkd --model resnet18 --custom_loss pearson --start_epoch_rate $SET --temperature 1.0 --lr 0.1 --dataset cifar100
        done
    done
done