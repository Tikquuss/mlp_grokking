#!/bin/bash

none="_None_"

### usage ###
# . train.sh $train_pct $weight_decay $representation_lr $decoder_lr $representation_dropout $decoder_dropout $opt $random_seed $operator $modular $p $task

### Main parameters ###
train_pct=${1-80}
weight_decay=${2-0.0}
representation_lr=${3-0.001}
decoder_lr=${4-0.001}
representation_dropout=${5-0.0}
decoder_dropout=${6-0.0}
opt=${7-adam}
random_seed=${8-0}

### Data parameters
operator=${9-+}
modular=${10-False}
p=${11-100}

### Task ### 
task=${12-classification}
#task=${12-regression}

## Other parameters
log_dir="../log_files"
max_epochs=10000
lr_scheduler=$none
#lr_scheduler=reduce_lr_on_plateau,factor=0.2,patience=20,min_lr=0.00005,mode=min,monitor=val_loss

#ID_params="method=str(twonn)"
ID_params=$none

### wandb ###
use_wandb=False
group_name="tdf=${train_pct}-wd=${weight_decay}-r_lr=${representation_lr}-d_lr=${decoder_lr}-r_d=${representation_dropout}-d_d=${decoder_dropout}-opt=${opt}"
wandb_entity="grokking_ppsp"
wandb_project="toy_model_grokking_op=${operator}-p=${p}-task=${task}-mod=${modular}"

#exp_id=$task
exp_id="${task}_${group_name}"

### Early_stopping (for grokking) : Stop the training `patience` epochs after the `metric` has reached the value `metric_threshold` ###
#early_stopping_grokking=$none
early_stopping_grokking="patience=int(1000),metric=str(val_acc),metric_threshold=float(90.0)"

python train.py \
	--task $task \
	--exp_id $exp_id \
	--log_dir "${log_dir}/${random_seed}" \
	--emb_dim 256 \
	--hidden_dim 512 \
	--n_layers 1 \
	--representation_dropout $representation_dropout \
	--decoder_dropout $decoder_dropout \
	--pad_index $none \
	--p $p \
	--operator $operator \
	--modular $modular \
	--train_pct $train_pct \
	--batch_size 512 \
	--optimizer "${opt},weight_decay=${weight_decay},beta1=0.9,beta2=0.99,eps=0.00000001" \
	--representation_lr $representation_lr \
	--decoder_lr $representation_lr \
	--lr_scheduler $lr_scheduler \
	--max_epochs $max_epochs \
	--validation_metrics val_loss \
	--checkpoint_path $none \
	--every_n_epochs 100 \
	--every_n_epochs_show 200 \
	--save_top_k -1 \
	--use_wandb $use_wandb \
	--wandb_entity $wandb_entity \
	--wandb_project $wandb_project \
	--group_name $group_name \
	--group_vars $none \
	--ID_params $ID_params \
	--accelerator auto \
	--devices auto \
	--random_seed $random_seed \
	--early_stopping_grokking $early_stopping_grokking \
#	--early_stopping_patience 1000000000 \
#	--model_name epoch=88-val_loss=13.6392.ckpt \

#filename=train.sh 
#cat $filename | tr -d '\r' > $filename.new && rm $filename && mv $filename.new $filename 