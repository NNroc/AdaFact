export PYTHONPATH="/data/npy/code/mrag:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python ./src/index.py --dataset "MMLong" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
python ./src/index.py --dataset "LongDocURL" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
python ./src/index.py --dataset "FetaTab" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
python ./src/index.py --dataset "PaperTab" --MM_ENCODER_MODEL "colqwen2.5-v0.2"
