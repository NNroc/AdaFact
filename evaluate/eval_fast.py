import ast
import os
import sys
import re
import json
import string
import numpy as np
from collections import Counter
from tqdm import tqdm
import argparse


def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.strip().split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output_info, labeled_answer, mode='qa'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0, 'hallucination': 0}

    pred_answer = output_info
    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            # acc1 = int(normalized_ground_truth in normalized_pred_answer) # 顺序也一样有些问题

            normalized_gt_set = set(normalized_ground_truth.split())
            normalized_pred_set = set(normalized_pred_answer.split())
            acc = int(normalized_gt_set.issubset(normalized_pred_set))

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)

            f1 = (2 * precision * recall) / (precision + recall + 1e-7)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    return final_metric, pred_answer


trn_set = set()


def run_evaluation(pred_list, gold_list, is_sufficient_list):
    # Existing evaluation for other datasets
    unsuff_em = []
    issuff_em, issuff_acc, issuff_f1 = [], [], []
    global_em, global_acc, global_f1 = [], [], []
    # avg_em, avg_acc, avg_f1, avg_hallucination, avg_bertscore = [], [], [], [], []

    i = 0
    for pred_answer, gold_answer, is_sufficient in tqdm(zip(pred_list, gold_list, is_sufficient_list)):

        if type(pred_answer) == str:
            if pred_answer.find("<answer>") != -1:
                pred_answer = pred_answer.split("<answer>")[1].split('</answer>')[0]
            output_text = pred_answer
        else:
            output_text = pred_answer.outputs[0].text

        if output_text in ['insufficient to answer', 'Not answerable']:
            output_text = 'Not answerable'
        if gold_answer in ['insufficient to answer', 'Not answerable']:
            gold_answer = 'Not answerable'

        metric, pred_answer = evaluate_predictions(output_text, gold_answer)

        if metric['acc'] == 1.0:
            trn_set.add(i)
        i += 1

        # Compute overall metrics
        global_em.append(metric['em'])
        global_acc.append(metric['acc'])
        global_f1.append(metric['f1'])
        # if is_sufficient:
        #     issuff_em.append(metric['em'])
        #     issuff_acc.append(metric['acc'])
        #     issuff_f1.append(metric['f1'])
        # else:
        #     unsuff_em.append(metric['em'])

    overall_results = {
        'global_em': np.mean(global_em) if len(global_em) > 0 else 0.0,
        'global_acc': np.mean(global_acc) if len(global_acc) > 0 else 0.0,
        'global_f1': np.mean(global_f1) if len(global_f1) > 0 else 0.0,
        # 'issuff_em': np.mean(issuff_em) if len(issuff_em) > 0 else 0.0,
        # 'issuff_acc': np.mean(issuff_acc) if len(issuff_acc) > 0 else 0.0,
        # 'issuff_f1': np.mean(issuff_f1) if len(issuff_f1) > 0 else 0.0,
        # 'unsuff_em': np.mean(unsuff_em) if len(unsuff_em) > 0 else 0.0,
        'cnt_global': len(global_em),
        'cnt_issuff': len(issuff_em),
        'cnt_unsuff': len(unsuff_em)
    }

    print(overall_results)
    return overall_results


def simple_str_to_list(s: str):
    s_stripped = s.strip()

    # 1. 如果字符串以 '[' 开头，尝试安全解析
    if s_stripped.startswith('['):
        try:
            # 尝试将字符串解析为 Python 字面量
            parsed = ast.literal_eval(s_stripped)

            # 确保解析结果是列表，否则视为解析失败
            if isinstance(parsed, list):
                return [str(item) for item in parsed]

        except (ValueError, SyntaxError):
            # 解析失败，可能是格式错误或不是合法的列表字面量
            # 2. 如果解析失败，回退到默认处理
            pass

    # 3. 默认处理：将原字符串作为单个元素放入列表中
    # 包括解析失败的情况、不以 '[' 开头的情况、以及空字符串
    return [s]


parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str)

args = parser.parse_args()


pred_ansewr_list, gold_answer_list, is_sufficient_list = [], [], []
with open(args.pred_path, 'r') as fin:
    items = json.loads(fin.read())
    for item_id, item_values in items.items():
        doc_id, answer, evidence_pages, evidence_sources, raw_response, pages_ranking = \
            item_values['doc_id'], item_values['answer'], item_values['evidence_pages'], \
                item_values['evidence_sources'], item_values['raw_response'], item_values['pages_ranking']

        is_sufficient = True

        pred_ansewr_list.append(raw_response)
        gold_answer_list.append(simple_str_to_list(answer))
        is_sufficient_list.append(is_sufficient)

overall_results = run_evaluation(pred_ansewr_list, gold_answer_list, is_sufficient_list)
