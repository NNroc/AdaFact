# shellcheck disable=SC2129
# shellcheck disable=SC2086
#set -x
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="/data/npy/code/AdaFact:$PYTHONPATH"
ulimit -n 8192

basepath="/data/npy/output"
eval_model="Qwen3-30B-A3B-Instruct-2507"
eval_url="http://127.0.0.1:8000/v1"
retrieve_method="colqwen"
top_k=5

model_names=("AdaFact-3B")
datasets=("MMLong" "PaperTab" "FetaTab" "LongDocURL")
prompt_methods=("ism")

bash ./script/evaluate_vllm.sh $eval_model
while true; do
  response=$(curl -s "${eval_url}/models")
  if [[ -n "$response" ]] && [[ "$response" == *"$eval_model"* ]]; then
      echo "Service is up and model $eval_model is loaded!"
      break
  else
      echo "Waiting for model $eval_model to load... (Response length: ${#response})"
      sleep 10
  fi
done
echo "vllm ${eval_model} 启动成功"

for model_name in "${model_names[@]}"; do
  model_name="${model_name//\//-}"
  for dataset in "${datasets[@]}"; do
    for prompt_method in "${prompt_methods[@]}"; do
      for ((i=1; i<=1; i++)); do
        python ./evaluate/eval_qa.py --dataset $dataset \
          --filepath "${basepath}/results/answer_results/${model_name}/${dataset}_${retrieve_method}_${top_k}_${prompt_method}_results.json" \
          --MODEL $eval_model --URL $eval_url >> "./log/eval_${dataset}.log"
      done
    done
  done
done

#pkill -u npy -f "VLLM::EngineCore"