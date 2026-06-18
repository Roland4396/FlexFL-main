#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FlexFL-main

python batch_train_all_vgg_resnet.py \
  --gpu 0 \
  --backbones vit \
  --datasets cifar10 \
  --methods FedAvg FedProx HeteroFL ScaleFL Decoupled FlexFL \
  --configs noniid_beta1 noniid_beta100 \
  --client-ratio 4:3:2:1 \
  --max-parallel 4 \
  --epochs 400 \
  --cpu-threads-per-job 3 \
  --output-dir outputs_batch_fill_vit_cifar10

python batch_train_all_vgg_resnet.py \
  --gpu 0 \
  --backbones vit \
  --datasets cifar100 \
  --methods FedAvg FedProx HeteroFL ScaleFL Decoupled FlexFL \
  --configs noniid_beta100 \
  --client-ratio 4:3:2:1 \
  --max-parallel 4 \
  --epochs 400 \
  --cpu-threads-per-job 3 \
  --output-dir outputs_batch_fill_vit_cifar100_alpha100

python batch_train_all_vgg_resnet.py \
  --gpu 0 \
  --backbones vit \
  --datasets TinyImagenet \
  --methods FedAvg FedProx HeteroFL ScaleFL Decoupled FlexFL \
  --configs noniid_beta1 noniid_beta100 \
  --client-ratio 4:3:2:1 \
  --max-parallel 4 \
  --epochs 400 \
  --cpu-threads-per-job 3 \
  --output-dir outputs_batch_fill_vit_tiny
