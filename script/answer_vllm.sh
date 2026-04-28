#!/bin/bash
export VLLM_LOGGING_LEVEL=ERROR
model_name=$1
MODEL_PATH="/data/npy/output/models/${model_name}"
echo $MODEL_PATH
# 配置三个实例的具体参数：GPU编号、端口、模型别名
# 格式: "GPU_ID:PORT:ALIAS"
model_name="${model_name//\//-}"
INSTANCES=(
  "0:8001:${model_name}"
  "1:8002:${model_name}"
)

for INSTANCE in "${INSTANCES[@]}"; do
    IFS=":" read -r GPU PORT ALIAS <<< "$INSTANCE"

    echo "正在启动实例: $ALIAS (GPU: $GPU, Port: $PORT)"

    # 使用 nohup 后台运行，并重定向日志
    CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$ALIAS" \
        --mm-processor-cache-gb 0 \
        --disable-mm-preprocessor-cache \
        --trust-remote-code \
        --gpu-memory-utilization 0.6 \
        --max-model-len 32768 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 128 \
        --swap-space 0 \
        --limit-mm-per-prompt '{"video": 0}' \
        --host 0.0.0.0 \
        --port "$PORT" > "./log/${PORT}.log" 2>&1 &

    echo "实例 $ALIAS 已在后台启动，日志记录在 ${PORT}.log"
done

echo "所有实例启动尝试完毕。可以使用 'ps -ef | grep vllm' 查看状态。"
