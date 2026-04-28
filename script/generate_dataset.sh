export PYTHONPATH="/data/npy/code/AdaFact:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
ulimit -n 16384

bash script/generate_vllm.sh
python ./dataset/generate_dataset_MPDocVQA.py --num_demo 15000 2>&1 | tee ./log/generate.log
python ./dataset/generate_dataset_SlideVQA.py --num_demo 35000 2>&1 | tee ./log/generate.log\
python ./dataset/process_dataset.py
pkill -u npy -f "VLLM::EngineCore"
