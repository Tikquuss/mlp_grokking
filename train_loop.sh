#!/bin/bash

# Usage : ./train_loop.sh

operator=+
modular=False
p=100
task=classification
#task=regression

for train_pct in 80; do {
for weight_decay in 0.0; do {
for representation_lr in 0.001; do {
for decoder_lr in 0.001; do {
for representation_dropout in 0.0; do {
for decoder_dropout in 0.0; do {
for opt in adam; do {
for random_seed in 0 100; do {
#. train.sh $train_pct $weight_decay $representation_lr $decoder_lr $representation_dropout $decoder_dropout $opt $random_seed
. train.sh $train_pct $weight_decay $representation_lr $decoder_lr $representation_dropout $decoder_dropout $opt $random_seed $operator $modular $p $task
} done
} done
} done
} done
} done
} done
} done
} done
