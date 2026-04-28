export PYTHONPATH="/data/npy/code/mrag:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python ./src/retrieve.py --dataset "MMLong" --method "colqwen" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
python ./src/retrieve.py --dataset "LongDocURL" --method "colqwen" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
python ./src/retrieve.py --dataset "FetaTab" --method "colqwen" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
python ./src/retrieve.py --dataset "PaperTab" --method "colqwen" --MM_ENCODER_MODEL "colqwen2.5-v0.2"

# python ./evaluate/eval_rag.py --dataset "MMLong" --filepath '/data/npy/output/results/retrieve_results/MMLong_colqwen_retrieval_results_eval.json'
