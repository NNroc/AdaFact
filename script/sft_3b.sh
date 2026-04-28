export CUDA_VISIBLE_DEVICES=0,1

epoch_num=2
dataset_dir=/data/npy/output/datasets/mrag-train
lr=1e-5
model_name=AdaFact-3B-sft
model_path=/data/npy/output/models/${model_name}

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /data/npy/models/Qwen/Qwen2.5-VL-3B-Instruct \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 4 \
    --finetuning_type full \
    --freeze_vision_tower true \
    --template qwen2_vl \
    --flash_attn fa2 \
    --dataset_dir ${dataset_dir} \
    --dataset mrag-train \
    --val_size 0.0 \
    --image_max_pixels 3920000 \
    --cutoff_len 31000 \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch_num} \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 100 \
    --warmup_ratio 0.1 \
    --packing False \
    --output_dir ${model_path} \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 3600000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --overwrite_output_dir True \
    --report_to tensorboard 2>&1 | tee -a ./${model_name}.log
