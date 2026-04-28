import os
import argparse
import json
import numpy as np
import prettytable as pt

from math import log2


def ndcg_cell(ground_truth, prediction, k):
    k = min(len(prediction), len(ground_truth), k)
    dcg = 0.0

    for i, doc_id in enumerate(prediction[:k]):
        rel = 1.0 if doc_id in ground_truth else 0.0
        dcg += rel / log2(i + 2)

    idcg = sum(1.0 / log2(i + 2) for i in range(k))

    if idcg == 0:
        return 0.0

    return dcg / idcg * 100.0


def mrr_cell(ground_truth, prediction, k):
    for i, item in enumerate(prediction[:k]):
        if item in ground_truth:
            return (1.0 / (i + 1)) * 100.0

    return 0.0


def evaluate_rag_one_sample(support_context, pred_context, top_k=[1, 5, 10]):
    """Evaluate the RAG on one sample
    Args:
        support_context (list): The ground truth evidence pages
        pred_context (list): The predicted evidence pages
        top_k (list): The top k to evaluate"""
    metrics = {}

    for k in top_k:
        cur_pred = pred_context[:k]
        intersect = len(set(cur_pred) & set(support_context))
        # Precision-related 
        metrics[f"recall@{k}"] = intersect / len(support_context) * 100.0
        metrics[f"precision@{k}"] = intersect / len(cur_pred) * 100.0 if len(cur_pred) > 0 else 0.0
        metrics[f"irrelevant@{k}"] = (len(cur_pred) - intersect) / len(cur_pred) * 100.0 if len(cur_pred) > 0 else 0.0

        # Ranking-related
        metrics[f"ndcg@{k}"] = ndcg_cell(support_context, cur_pred, k)
        metrics[f"mrr@{k}"] = mrr_cell(support_context, cur_pred, k)

    return metrics


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # only MMLong and LongDocURL provide the ground-truth evidence pages
    args.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL"])
    args.add_argument("--k_list", type=list, default=[1, 3, 5, 10])
    args.add_argument("--filepath", type=str, default='')
    args = args.parse_args()

    # 定义证据类型
    if args.dataset == "MMLong":
        evidence_types = ['Table', 'Figure', 'Chart', 'Pure-text (Plain-text)', 'Generalized-text (Layout)']
    elif args.dataset == "LongDocURL":
        evidence_types = ['Table', 'Figure', 'Text', 'Layout', 'Others']

    for retrieve_method in ["base"]:
        filepath = args.filepath
        if not os.path.exists(filepath):
            continue

        table = pt.PrettyTable()
        table.field_names = ["Method", "K", "Recall(%)", "Precision(%)", "NDCG(%)", "MRR(%)", "Irrelevant(%)"]

        # 创建按类型分类的指标存储
        type_tables = {}
        type_metrics = {}
        type_sample_counts = {}  # 新增：存储每种类型的样本数量
        for etype in evidence_types:
            type_tables[etype] = pt.PrettyTable()
            type_tables[etype].field_names = ["Method", "K", "Recall(%)", "Precision(%)", "NDCG(%)", "MRR(%)",
                                              "Irrelevant(%)", "Sample_Count"]
            type_metrics[etype] = {f'recall@{k}': [] for k in args.k_list}
            for remain_metric in ['ndcg', 'mrr', 'precision', 'irrelevant']:
                type_metrics[etype].update({f'{remain_metric}@{k}': [] for k in args.k_list})
            type_sample_counts[etype] = 0  # 初始化样本计数

        samples = json.load(open(filepath, 'r'))
        all_metrics = {f'recall@{k}': [] for k in args.k_list}
        for remain_metric in ['ndcg', 'mrr', 'precision', 'irrelevant']:
            all_metrics.update({f'{remain_metric}@{k}': [] for k in args.k_list})

        for sample in samples:
            if eval(sample["evidence_pages"]) == [] or "pages_ranking" not in sample:
                continue

            preds = eval(sample["pages_ranking"])
            score = evaluate_rag_one_sample(
                support_context=eval(sample["evidence_pages"]),
                pred_context=preds,
                top_k=args.k_list
            )

            for metric_name, value in score.items():
                all_metrics[metric_name].append(value)

            # 新增：按证据类型分类统计
            if "evidence_sources" in sample:
                sample_evidence_sources = eval(sample["evidence_sources"])
                for etype in evidence_types:
                    if etype in sample_evidence_sources:
                        for metric_name, value in score.items():
                            type_metrics[etype][metric_name].append(value)
                        type_sample_counts[etype] += 1  # 计数

        # 原有总体统计
        for metric_name, values in all_metrics.items():
            all_metrics[metric_name] = np.round(np.mean(values), 2)

        for k in args.k_list:
            table.add_row([retrieve_method, k, all_metrics[f'recall@{k}'], all_metrics[f'precision@{k}'],
                           all_metrics[f'ndcg@{k}'], all_metrics[f'mrr@{k}'], all_metrics[f'irrelevant@{k}']])
        print("Overall Results:")
        print(table, '\n')

        # 新增：打印按类型分类的结果
        print("Breakdown by Evidence Type:")
        for etype in evidence_types:
            sample_count = type_sample_counts[etype]
            if sample_count > 0:  # 如果有该类型的样本
                # 计算该类型的平均指标
                type_avg_metrics = {}
                for metric_name, values in type_metrics[etype].items():
                    if values:  # 确保列表不为空
                        type_avg_metrics[metric_name] = np.round(np.mean(values), 2)
                    else:
                        type_avg_metrics[metric_name] = 0.0

                for k in args.k_list:
                    type_tables[etype].add_row([
                        retrieve_method, k,
                        type_avg_metrics[f'recall@{k}'],
                        type_avg_metrics[f'precision@{k}'],
                        type_avg_metrics[f'ndcg@{k}'],
                        type_avg_metrics[f'mrr@{k}'],
                        type_avg_metrics[f'irrelevant@{k}'],
                        sample_count
                    ])
                print(f"{etype} Evidence Results (n={sample_count}):")
                print(type_tables[etype], '\n')
