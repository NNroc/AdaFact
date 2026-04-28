import ast
import os
import copy
import pickle
import random
import argparse
import sys
import traceback
import logging

# 禁用所有 httpx 的日志（vLLM 客户端常用）
logging.getLogger("httpx").setLevel(logging.WARNING)
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from config.base_config import global_config
from generate.adafact.prompt import PROMPTS
from generate.adafact.generate_utils import run_generate_process
from llm.vllm import multimodel_if_cache
from utils.file_utils import write_json, load_json
from utils.storage import JsonKVStorage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--save_file", type=str, default='/data/npy/output/datasets/mrag-test')
    p.add_argument("--num_demo", type=int, default=2000)
    return p.parse_args()


# 获取训练集的问题、答案、参考文件和参考文件的证据页，对于每个问题，生成5个不同的证据页组合，因为使用不同的证据页进行问答可以得到不同的输出结果，其中4个组合都要包含完整的证据,证据的位置均匀分布，我需要通过这种办法构建训练数据集
if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    global_config.MM_MODEL = 'Qwen3-VL-30B-A3B-Instruct'
    save_file_path = args.save_file
    os.makedirs(save_file_path, exist_ok=True)

    test_path = f'/data/npy/output/results/retrieve_results/MMLong_colqwen_retrieval_results_eval.json'
    test_samples = load_json(test_path)
    all_combinations = []
    for sample in test_samples:
        retrieve_pages = ast.literal_eval(test_samples[sample]['pages_ranking'])[:5]
        retrieve_information = []
        image_path = test_samples[sample]['doc_id'].split('.pdf')[0]
        for ri in retrieve_pages:
            retrieve_information.append({
                'input_page_idx': ri,
                'input_path': f"imgs/{image_path}-{str(ri)}.png",
                'input_type': 'image',
            })
        all_combinations.append({
            'question': test_samples[sample]['question'],
            'answer': test_samples[sample]['answer'],
            'evidence_pages':[],
            'retrieve_information': retrieve_information,
        })

    for combination in all_combinations:
        combination['prompt'] = PROMPTS["question_answer"].format(query=combination['question'])

        combination['use_retrieve_information'] = []
        re_use = copy.deepcopy(combination['retrieve_information'])
        for ri_idx, ri in enumerate(re_use):
            ri['input_path'] = f'{save_file_path}/' + ri['input_path']
            ri['input_id'] = f'\nImage {str(ri_idx + 1)}'
            combination['use_retrieve_information'].append(ri)

    use_combinations = all_combinations[:args.num_demo]

    run_generate_process(use_combinations, save_file_path)


