import os
import copy
import pickle
import random
import argparse
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm

from config.base_config import global_config
from generate.mrag_dataset.prompt import PROMPTS
from generate.mrag_dataset.generate_utils import run_generate_process


def collect_page_texts(example_row: Dict[str, Any]) -> Dict[int, str]:
    """
    从一条 dataset row 中收集 page_1..page_20 的文本（如果存在）。
    返回 dict: {page_idx (1-based): page_text}
    """
    texts = {}
    for i in range(1, 21):
        key = f"page_{i}"
        if key in example_row and example_row[key] is not None:
            # 页面字段可能包含图片标签或 OCR 文本，直接保存原始字段内容
            texts[i] = example_row[key]
    return texts


def parse_evidence_pages_field(evidence_field: Any) -> List[int]:
    """
    尝试从 evidence_pages 字段解析出页码列表（假设为 list[int] 或用逗号分隔字符串等）。
    返回整数页码列表（1-based）。
    """
    if evidence_field is None:
        return []
    if isinstance(evidence_field, list):
        return [int(x) for x in evidence_field if x is not None]
    if isinstance(evidence_field, str):
        s = evidence_field.strip()
        if not s:
            return []
        # 可能格式如 "1,2,3" 或 "[1,2]"
        s = s.replace("[", "").replace("]", "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        out = []
        for p in parts:
            try:
                out.append(int(p))
            except:
                # 忽略非数字
                pass
        return out
    # 其他情况
    try:
        return [int(evidence_field)]
    except:
        return []


def build_evidence_combinations(all_pages: List[int],
                                evidence_pages: List[int],
                                num_combinations: int = 5,
                                num_combin_page: int = 5) -> List[List[int]]:
    # 1. 预处理：去重并保持顺序
    all_pages_set = list(dict.fromkeys(all_pages))
    if not all_pages_set:
        return []

    # 限制每组页面数不能超过总页面数
    num_combin_page = min(num_combin_page, len(all_pages_set))

    # 过滤证据页，确保都在 all_pages 中
    evidence_pages = [p for p in evidence_pages if p in all_pages_set]
    non_evidence_pages = [p for p in all_pages_set if p not in evidence_pages]

    combos_result = []
    seen_sets = []  # 用于记录已生成的组合内容（无序比较）

    # 2. 生成组合逻辑
    for i in range(num_combinations):
        is_full_evidence = i < (num_combinations - 1)
        combo = []

        if is_full_evidence and evidence_pages:
            # 策略 A: 尽量包含所有证据
            combo.extend(evidence_pages)
        elif evidence_pages:
            # 策略 B: 包含部分证据 (随机取一半左右)
            k = len(evidence_pages) // 2
            combo.extend(random.sample(evidence_pages, k))

        # 3. 填充与裁剪
        if len(combo) > num_combin_page:
            # 如果证据页本身太多，随机抽样保留
            combo = random.sample(combo, num_combin_page)
        else:
            # 填充剩余位置：从非证据页中随机抽取
            needed = num_combin_page - len(combo)
            if len(non_evidence_pages) >= needed:
                combo.extend(random.sample(non_evidence_pages, needed))
            else:
                # 理论上不会走到这里，因为 num_combin_page 已被 min 限制
                combo.extend(non_evidence_pages)

        # 4. 强力去重逻辑
        # 如果当前组合已存在，尝试随机替换一个非证据元素
        attempts = 0
        current_set = set(combo)
        while current_set in seen_sets and attempts < 10:
            idx_to_replace = random.randrange(len(combo))
            # 尝试换成一个目前不在组合里的页面
            pool = [p for p in all_pages_set if p not in current_set]
            if pool:
                combo[idx_to_replace] = random.choice(pool)
                current_set = set(combo)
            attempts += 1

        random.shuffle(combo)
        combos_result.append(combo)
        seen_sets.append(current_set)

    return combos_result



def save_page_image(page_texts, all_pages, output_path, deck_name):
    os.makedirs(f'{output_path}/imgs/slidevqa', exist_ok=True)
    for page_idx in all_pages:
        page_pil = page_texts.get(page_idx)
        file_name = f"{deck_name}-{page_idx}.png"
        absolute_path = Path(output_path, 'imgs', 'slidevqa', file_name)
        if not absolute_path.exists():
            page_pil.save(absolute_path)


def assemble_retrieve_information(page_texts, pages_order, deck_name) -> List[Dict[str, Any]]:
    """
    将 page_texts（dict: page_idx -> text）和 pages_order 转成模型检索信息结构。
    """
    out = []
    for idx, page_idx in enumerate(pages_order):
        page_pil = page_texts[page_idx]
        page_path = f'slidevqa/{deck_name}-{page_idx}.png'
        out.append({
            "page_type": 'image',
            "page_idx": page_idx,
            "image_path": page_path,
            "page_pil": page_pil
        })
    return out


def process_slidevqa_and_generate(output_path: str,
                                  slidevqa_path_pattern: str = '/data/npy/datasets/SlideVQA/data/train-*.parquet',
                                  num_combinations: int = 5,
                                  num_combin_page: int = 5,
                                  start_index: int = 0,
                                  cache_path: Optional[str] = None):
    # 🌟 缓存读取逻辑
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Found cache file at {cache_path}. Loading data from cache.")
        try:
            with open(cache_path, 'rb') as f:
                all_combinations = pickle.load(f)
            print("[INFO] Successfully loaded data from cache.")
            return all_combinations
        except Exception as e:
            print(f"[WARN] Failed to load cache file: {e}. Proceeding with data generation.")
            # 如果加载失败，继续执行生成逻辑
            pass
    """
    主函数：加载数据、构造 combos、调用 LLM、写入输出文件
    """
    os.makedirs(f'{output_path}/imgs', exist_ok=True)

    # Load dataset using datasets.load_dataset
    data_files = {'train': slidevqa_path_pattern}
    print(f"[INFO] Loading SlideVQA dataset from pattern: {slidevqa_path_pattern}")
    dataset = load_dataset('parquet', data_files=data_files)
    train_set = dataset['train']
    total_rows = len(train_set)
    print(f"[INFO] dataset loaded. Train rows: {total_rows}")

    iterator = range(start_index, total_rows)

    all_combinations = []
    for idx in tqdm(iterator, desc="Processing combinations..."):
        try:
            row = train_set[idx]
            deck_name = row.get('deck_name')
            qa_id = f"train_{idx}"
            question = row.get("question", "") or ""
            answer = row.get("answer", "") or ""
            evidence_field = row.get("evidence_pages", None)
            evidence_pages = parse_evidence_pages_field(evidence_field)
            page_texts = collect_page_texts(row)
            all_pages = sorted(list(page_texts.keys()))
            save_page_image(page_texts, all_pages, '/data/npy/output/datasets/mrag-train', deck_name)

            if not all_pages:
                print(f"[WARN] no page fields found for index {idx}, qa_id {qa_id}. Skipping.")
                continue
            combos = build_evidence_combinations(all_pages, evidence_pages,
                                                 num_combinations=num_combinations, num_combin_page=num_combin_page)

            for combo_idx, combo_pages in enumerate(combos):
                retrieve_info = assemble_retrieve_information(page_texts, combo_pages, deck_name)
                # Format the prompt by injecting query and examples
                page_idx = list()
                retrieve_information = []
                for retrieve_result in retrieve_info:
                    information = {}
                    page_idx.append(retrieve_result['page_idx'])
                    information['input_page_idx'] = retrieve_result['page_idx']
                    information['input_path'] = f"imgs/{retrieve_result['image_path']}"
                    information['input_type'] = retrieve_result['page_type']
                    retrieve_information.append(information)

                all_combinations.append({
                    "question": question,
                    "answer": answer,
                    "evidence_pages": evidence_pages,
                    "retrieve_information": retrieve_information,
                })
        except Exception as e:
            print(f"[ERROR] processing index {idx} failed: {e}")
            continue

    if cache_path:
        print(f"[INFO] Saving results to cache file at {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(all_combinations, f)
            print("[INFO] Successfully saved data to cache.")
        except Exception as e:
            print(f"[WARN] Failed to save cache file: {e}")

    return all_combinations


def parse_args():
    p = argparse.ArgumentParser(description="Build MRAG train dataset from SlideVQA via LLM prompts.")
    p.add_argument("--slidevqa_pattern", type=str, default='/data/npy/datasets/SlideVQA/data/train-*.parquet')
    p.add_argument("--save_file", type=str, default='/data/npy/output/datasets/mrag-train/v13/SlideVQA')
    p.add_argument("--pages_per_combo", type=int, default=5, help="每个组合包含多少页")
    p.add_argument("--num_demo", type=int, default=50)
    return p.parse_args()


# 获取训练集的问题、答案、参考文件和参考文件的证据页，对于每个问题，生成5个不同的证据页组合，因为使用不同的证据页进行问答可以得到不同的输出结果，其中4个组合都要包含完整的证据,证据的位置均匀分布，我需要通过这种办法构建训练数据集
if __name__ == "__main__":
    args = parse_args()
    random.seed(42)
    global_config.MM_MODEL = 'Qwen3-VL-30B-A3B-Instruct'
    global_config.MM_MODEL = 'Qwen3-VL-32B-Instruct'
    save_file_path = args.save_file
    os.makedirs(save_file_path, exist_ok=True)

    cache_file = f'{save_file_path}/all_combinations_cache.pkl'
    all_combinations = process_slidevqa_and_generate(output_path=f'{save_file_path}',
                                                     slidevqa_path_pattern=args.slidevqa_pattern,
                                                     cache_path=cache_file)

    group_size = 5  # 周期大小 10617
    keep_limit = [0, 1, 4]
    question_counts = defaultdict(int)
    filtered_combinations = []
    for item in all_combinations:
        question = item.get("question")
        current_count = question_counts[question]
        if current_count % group_size in keep_limit:
            if current_count % group_size == 4:
                item["answer"] = "Not answerable"
            filtered_combinations.append(item)
        question_counts[question] += 1
    all_combinations = filtered_combinations

    use_combinations = all_combinations[:args.num_demo]

    for combination in use_combinations:
        combination['prompt'] = PROMPTS["question_answer"].format(query=combination['question'])

        combination['use_retrieve_information'] = []
        re_use = copy.deepcopy(combination['retrieve_information'])
        for ri_idx, ri in enumerate(re_use):
            ri['input_path'] = f'/data/npy/output/datasets/mrag-train/{ri['input_path']}'
            ri['input_id'] = f'Image {str(ri_idx + 1)}'
            combination['use_retrieve_information'].append(ri)

    run_generate_process(use_combinations,  save_file_path)
