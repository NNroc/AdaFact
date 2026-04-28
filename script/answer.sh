# shellcheck disable=SC2129
# shellcheck disable=SC2086
export PYTHONPATH="/data/npy/code/AdaFact:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
ulimit -n 8192

port=8000
mm_url="http://127.0.0.1:${port}/v1"
retrieve_method="colqwen"
top_ks=(5)

model_names=("AdaFact-3B")
datasets=("MMLong" "PaperTab" "FetaTab" "LongDocURL")
prompt_methods=("ism")


for model_name in "${model_names[@]}"; do
  bash ./script/answer_vllm.sh $model_name
  model_name="${model_name//\//-}"
  while true; do
    response=$(curl -s "${mm_url}/models")
    if [[ -n "$response" ]] && [[ "$response" == *"$model_name"* ]]; then
        echo "Service is up and model $model_name is loaded!"
        break
    else
        sleep 10
    fi
  done
  echo "vllm ${model_name} start!"

  for dataset in "${datasets[@]}"; do
    for prompt_method in "${prompt_methods[@]}"; do
        for top_k in "${top_ks[@]}"; do
          python ./src/answer.py --dataset $dataset --prompt_method $prompt_method --retrieve_method $retrieve_method \
            --top_k $top_k --MM_MODEL $model_name --MM_URL $mm_url >> "./log/train_${dataset}.log"
        done
    done
  done
  pkill -u npy -f "VLLM::EngineCore"
  sleep 5
done

bash script/evaluate.sh
