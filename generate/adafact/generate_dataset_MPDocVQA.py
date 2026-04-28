import os
import copy
import random
import argparse
import logging

from PIL import Image


# 禁用所有 httpx 的日志（vLLM 客户端常用）
logging.getLogger("httpx").setLevel(logging.WARNING)
from pathlib import Path
from datasets import load_dataset

from config.base_config import global_config
from generate.mrag_dataset.prompt import PROMPTS
from generate.mrag_dataset.generate_utils import run_generate_process
from utils.file_utils import load_json


# 如果你希望保持和load_base64完全一致的逻辑，这里还有一个更接近的版本：
def save_page_image(page_idx, img, output_path, deck_name, max_side_pixels=1536):
    os.makedirs(f'{output_path}/imgs/mpdocvqa', exist_ok=True)
    file_name = f"{deck_name}-{page_idx}.png"
    absolute_path = Path(output_path, 'imgs', 'mpdocvqa', file_name)
    if not absolute_path.exists():
        # 复制一份原始图片进行处理
        width, height = img.size
        reduce = 1.0
        # 计算缩放比例
        if max(width, height) > max_side_pixels:
            reduce = max(width, height) / max_side_pixels
        # 如果需要缩放
        if reduce > 1.0:
            # 计算新尺寸
            new_size = (int(width // reduce), int(height // reduce))
            new_size = (max(1, new_size[0]), max(1, new_size[1]))
            # 缩放图片（使用高质量的重采样算法）
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        # 保存图片
        img.save(absolute_path, format="PNG", optimize=True)
        img.close()


def save_imgs(rows):
    for page_number, image, document_id in zip(rows['page_number'], rows['image'], rows['document_id']):
        save_page_image(page_number, image, '/data/npy/output/datasets/mrag-train', document_id)

def process_MPDocVQA_imgs(imgs_path: str):
    data_files = {'train': imgs_path}
    print(f"[INFO] Loading dataset from pattern: {imgs_path}")
    dataset = load_dataset('parquet', data_files=data_files)
    train_set = dataset['train']
    total_rows = len(train_set)
    print(f"[INFO] dataset loaded. Train rows: {total_rows}")
    # iterator = range(0, total_rows)
    # rows = []
    # for idx in tqdm(iterator):
    #     rows.append(train_set[idx])
    train_set.map(save_imgs, batched=True, batch_size=32, remove_columns=train_set.column_names, num_proc=48)

    print(f"[INFO] Images saved. Total rows: {total_rows}")


def process_MPDocVQA_train(train_data):
    use_data = []
    for item in train_data["data"]:
        page_count = len(item.get("page_ids", []))
        if 2 <= page_count <= 5:
            if "answers" in item and item["answers"]:
                longest_answer = max(item["answers"], key=len)
                item["answers"] = [longest_answer]
            else:
                continue
            page_idx = list()
            evidence_pages = []
            retrieve_information = []
            random.shuffle(item["page_ids"])
            for retrieve_result in item["page_ids"]:
                page_information = retrieve_result.split("_p")
                information = {}
                page_idx.append(int(page_information[1]))
                information['input_page_idx'] = int(page_information[1])
                information['input_path'] = f"imgs/mpdocvqa/{page_information[0]}-{page_information[1]}.png"
                information['input_type'] = "image"
                evidence_pages.append(int(page_information[1]))
                retrieve_information.append(information)
            use_data.append({
                "question": item["question"],
                "answer": longest_answer,
                "evidence_pages": evidence_pages,
                "retrieve_information": retrieve_information,
            })
    print(f"[INFO] Filtered {len(use_data)} pages")
    return use_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--MPDocVQA_pattern", type=str,
                   default='/data/npy/datasets/AHS-uni/mpdocvqa-corpus/data/train-*.parquet')
    p.add_argument("--save_file", type=str, default='/data/npy/output/datasets/mrag-train/v13/MPDocVQA')
    p.add_argument("--num_demo", type=int, default=100000)
    return p.parse_args()


# 获取训练集的问题、答案、参考文件和参考文件的证据页，对于每个问题，生成5个不同的证据页组合，因为使用不同的证据页进行问答可以得到不同的输出结果，其中4个组合都要包含完整的证据,证据的位置均匀分布，我需要通过这种办法构建训练数据集
if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    global_config.MM_MODEL = 'Qwen3-VL-30B-A3B-Instruct'
    global_config.MM_MODEL = 'Qwen3-VL-32B-Instruct'
    save_file_path = args.save_file
    os.makedirs(save_file_path, exist_ok=True)
    # process_MPDocVQA_imgs(imgs_path=f'{args.MPDocVQA_pattern}')
    # 读取训练集 12112
    train_data = load_json('/data/npy/datasets/AHS-uni/mpdocvqa-qa/data/train.json')
    all_combinations = process_MPDocVQA_train(train_data)

    use_combinations = all_combinations[:args.num_demo]

    for combination in use_combinations:
        combination['prompt'] = PROMPTS["question_answer"].format(query=combination['question'])

        combination['use_retrieve_information'] = []
        re_use = copy.deepcopy(combination['retrieve_information'])
        for ri_idx, ri in enumerate(re_use):
            ri['input_path'] = f'/data/npy/output/datasets/mrag-train/{ri['input_path']}'
            ri['input_id'] = f'Image {str(ri_idx + 1)}'
            combination['use_retrieve_information'].append(ri)

    run_generate_process(use_combinations, save_file_path)
