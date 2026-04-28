export VLLM_LOGGING_LEVEL=ERROR
model_name="Qwen3-VL-30B-A3B-Instruct"
model_path="/data/npy/models/Qwen/${model_name}"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model $model_path \
  --served-model-name $model_name \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 65536 \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --swap-space 0 \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --host 0.0.0.0  \
  --port 8011 > "./log/8011.log" 2>&1 &

#CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#  --model $model_path \
#  --served-model-name $model_name \
#  --mm-processor-cache-gb 0 \
#  --gpu-memory-utilization 0.95 \
#  --max-num-batched-tokens 65536 \
#  --max-model-len 65536 \
#  --max-num-seqs 128 \
#  --enable-prefix-caching \
#  --swap-space 0 \
#  --limit-mm-per-prompt '{"image":8,"video":0}' \
#  --host 0.0.0.0  \
#  --port 8012 > "./log/8012.log" 2>&1 &
