export VLLM_LOGGING_LEVEL=ERROR
model_name="Qwen3-30B-A3B-Instruct-2507"
model_path="/data/npy/models/Qwen/${model_name}"

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model $model_path \
  --served-model-name $model_name \
  --trust-remote-code \
  --quantization fp8 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 8000 \
  --max-model-len 8000 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --swap-space 0 \
  --host 0.0.0.0  \
  --port 8001 > "./log/${model_name}.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model $model_path \
  --served-model-name $model_name \
  --trust-remote-code \
  --quantization fp8 \
  --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 8000 \
  --max-model-len 8000 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --swap-space 0 \
  --host 0.0.0.0  \
  --port 8002 > "./log/${model_name}.log" 2>&1 &
