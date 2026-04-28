import argparse
import copy
import json
import os
import random
import re

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.file_utils import load_json, write_json
from utils.reward_model_utils import get_raw_score


def format_punishment(response: str) -> float:
    # 基础结构匹配：plan -> 至少一个record -> reason -> answer
    # 允许 record 标签内有属性，且内容必须以指定的 Judge 语句结尾
    pattern = r"<plan>.*?</plan>\s*(<record.*?>.*?</record>\s*)+<reason>.*?</reason>\s*<answer>.*?</answer>"
    if not re.fullmatch(pattern, response.strip(), flags=re.DOTALL):
        return -1.0

    # 2. 提取所有的 records 进行深度检查
    record_tags = re.findall(r"<record image=\"(\d+)\">(.*?)</record>", response, re.DOTALL)

    # 如果没匹配到带 image 属性的 record（或者格式不对），返回错误
    if not record_tags:
        return -1.0

    expected_index = 1
    for idx_str, content in record_tags:
        # 检查序号是否按顺序递增 (1, 2, 3...)
        if int(idx_str) != expected_index:
            return -1.0

        # 检查内容是否以指定的 Judge 结尾（忽略末尾空格）
        content_clean = content.strip()
        valid_judgments = ("Judge: Insufficient information", "Judge: Sufficient information")
        if not any(content_clean.endswith(j) for j in valid_judgments):
            return -1.0

        expected_index += 1

    # 3. 唯一性标签校验
    unique_tags = ["plan", "reason", "answer"]
    for tag in unique_tags:
        if len(re.findall(f"<{tag}>", response)) != 1: return 0.0

    # 4. Reason 与 Answer 的逻辑隔离
    reason_content = re.search(r"<reason>(.*?)</reason>", response, re.DOTALL).group(1).strip()
    answer_content = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL).group(1).strip()

    if reason_content == answer_content:
        return -1.0

    return 0.0


def clean_content(content: str) -> str | None:
    # 截取有效区间
    plan_start_match = re.search(r'<plan>', content, re.DOTALL)
    answer_end_matches = list(re.finditer(r'</answer>', content, re.DOTALL))

    if not plan_start_match or not answer_end_matches:
        return None

    start_index = plan_start_match.start()
    end_index = answer_end_matches[-1].end()
    cleaned_content = content[start_index:end_index].strip()

    # 将多个 \n 替换为单个 \n
    cleaned_content = re.sub(r'\n+', '\n', cleaned_content)

    # 清除不在标签前后的换行符
    # 逻辑：匹配 \n，但使用“负向环视”确保它前后不是 '>' 或 '<'
    # (?<!>) 表示前面不是 '>'
    # (?!<)  表示后面不是 '<'
    cleaned_content = re.sub(r'(?<!>)\n(?!<)', '', cleaned_content)

    return cleaned_content


def plot_token_distribution(lengths, save_path='prompt_dist.png'):
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=30, kde=True, color='skyblue')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path)
    print(f"统计图表已保存至: {save_path}")


def plot_rm_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=20, color='red', linestyle='--', label='Threshold (20)')
    plt.title('Distribution of RM Scores')
    plt.xlabel('Score Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('rm_score_distribution.png')
    print(f"Stats: Mean={np.mean(scores):.2f}, Median={np.median(scores):.2f}, Total={len(scores)}")


def no_answerable_proceed(sample):
    image_num = len(sample['images'])
    sample['messages'][0]['content'] = sample['messages'][0]['content'].replace("<image>", "") + "<image>" * image_num
    plan_content = sample['messages'][1]['content'].split('<plan>')[1].split('</plan>')[0]
    plan_tags = f"<plan>{plan_content}</plan>\n"
    records = []
    for i in range(1, image_num + 1):
        record = f'<record image="{i}">No relevant information. Judge: Insufficient information</record>\n'
        records.append(record)
    records_str = "".join(records)
    reason_tags = "<reason>No information relevant to the question was found in the provided input.</reason>\n"
    answer_tags = "<answer>Not answerable</answer>"
    final_content = f"{plan_tags}{records_str}{reason_tags}{answer_tags}"
    sample['messages'][1]['content'] = final_content
    return sample


def swap_multimodal_data(rl_results):
    unique_samples = {}
    for item in rl_results:
        try:
            content = item['messages'][0]['content']
            parts = content.split("Question:", 1)
            if len(parts) < 2: continue
            question = parts[1].split("\n")[0].strip()
            if question in unique_samples:
                continue

            assistant_content = item['messages'][1]['content']
            if "<answer>Not answerable</answer>" in assistant_content:
                if question in unique_samples:
                    del unique_samples[question]
                continue

            img_path = item['images'][0]
            prefix = re.sub(r'-\d+\.(png|jpg|jpeg)$', '', img_path)
            unique_samples[question] = {'data': item, 'prefix': prefix, 'question': question}
        except (IndexError, KeyError):
            continue

    samples_list = list(unique_samples.values())
    random.shuffle(samples_list)
    new_data_batch = []
    used_indices = set()

    for i in range(len(samples_list)):
        if i in used_indices:
            continue

        target_a = samples_list[i]

        for j in range(i + 1, len(samples_list)):
            if j in used_indices:
                continue

            target_b = samples_list[j]

            if target_a['prefix'] != target_b['prefix']:
                new_item_1 = copy.deepcopy(target_a['data'])
                new_item_1['images'] = copy.deepcopy(target_b['data']['images'])
                new_item_2 = copy.deepcopy(target_b['data'])
                new_item_2['images'] = copy.deepcopy(target_a['data']['images'])
                new_item_1 = no_answerable_proceed(new_item_1)
                new_item_2 = no_answerable_proceed(new_item_2)
                new_data_batch.extend([new_item_1, new_item_2])
                used_indices.add(i)
                used_indices.add(j)
                break

    return new_data_batch


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_file", type=str, default='/home/npy/output/datasets/mrag-train')
    p.add_argument("--filename", type=str, default='origin_dataset.json')
    args = p.parse_args()
    random.seed(42)
    prompt_lengths = []
    record_num_in_1, record_num_in_n, not_answerable_num_in_rl = 0, 0, 0
    answerable_num_in_1_rl, not_answerable_num_in_1_rl = 0, 0
    single_reason_num_in_rl, multi_reason_num_in_rl, single_reason_num_in_sft, multi_reason_num_in_sft = 0, 0, 0, 0
    all_scores, all_scores_prompt, all_scores_answer, existing_scores = [], [], [], {}
    if os.path.exists(f'{args.save_file}/scores_data.jsonl'):
        with open(f'{args.save_file}/scores_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                all_scores.append(data["score"])
                all_scores_prompt.append(data["prompt"])
                all_scores_answer.append(data["response"])
                key = (data["prompt"], data["response"])
                existing_scores[key] = data["score"]
        print(f"已加载 {len(all_scores)} 条现有数据。")
    else:
        print("未发现现有评分文件，将从零开始记录。")

    error_0, error_1, error_2, error_3, error_4 = 0, 0, 0, 0, 0
    if "mrag-train" in args.save_file:
        origin_results_slidevqa = load_json(f'{args.save_file}/SlideVQA/{args.filename}')
        origin_results_mpdocvqa = load_json(f'{args.save_file}/MPDocVQA/{args.filename}')
        origin_results = origin_results_slidevqa + origin_results_mpdocvqa
    elif "mrag-test" in args.save_file:
        origin_results = load_json(f'/data/npy/output/datasets/mrag-test/origin_dataset.json')
    else:
        raise NotImplementedError

    sft_results, rl_results, sft_flag, rl_flag = [], [], [], []
    chose_results_flag, cleaned_results = set(), []
    for origin_result in tqdm(origin_results):
        response = '\n'.join(origin_result['history'])

        if '<image>' in response or '</image>' in response or len(response) > 4000:
            error_1 += 1
            continue

        cleaned_content = clean_content(response)
        if cleaned_content is None:
            error_0 += 1
            continue
        if format_punishment(cleaned_content) != 0:
            error_0 += 1
            continue

        imgs = []
        images_token = 0
        prompt = origin_result['prompt'].split('##########')[0]
        prompt = prompt + f"\nQuestion: {origin_result['question']}\n"

        retrieve_information_len = len(origin_result["retrieve_information"])
        if retrieve_information_len > 5:
            continue
        record_len = origin_result['history'][1].count('</record>')
        origin_num = random.randint(record_len, retrieve_information_len)

        response_answer = response.split("<answer>")[1].split('</answer>')[0]

        if (response.count('Judge: Insufficient information') < retrieve_information_len
                and response.count('Judge: Sufficient information') == 0):
            error_2 += 1
            continue
        if 'Not answerable' in response_answer and 'Not answerable' not in origin_result["answer"]:
            error_3 += 1
            continue
        if 'Not answerable' in response_answer and record_len < retrieve_information_len:
            error_3 += 1
            continue
        if record_len < len(origin_result["evidence_pages"]) \
                and "/slidevqa/" in origin_result['retrieve_information'][0]['input_path']:
            error_3 += 1
            continue

        if "/slidevqa/" in origin_result['retrieve_information'][0]['input_path'] \
                and len(origin_result["evidence_pages"]) == 1 \
                and origin_result['question'] in sft_flag \
                and 'Not answerable' not in response_answer:
            continue
        else:
            sft_flag.append(origin_result['question'])

        # 判断回答的好不好
        match = re.search(r"<answer>(.*?)</answer>", response, re.S)
        answer = match.group(1).strip()
        if (prompt + f"\nThe golden answer is: {answer}", response) in existing_scores:
            rm_score = existing_scores[(prompt + f"\nThe golden answer is: {answer}", response)]
        else:
            rm_score = get_raw_score(prompt + f"\nThe golden answer is: {answer}", response)
            all_scores_prompt.append(prompt + f"\nThe golden answer is: {answer}")
            all_scores_answer.append(response)
            all_scores.append(rm_score)

        for ri_idx, ri in enumerate(origin_result['retrieve_information']):
            if ri_idx == origin_num:
                break
            if not ri['input_path'].endswith('.png'):
                ri['input_path'] += '.png'
            imgs.append(ri['input_path'])
            if ri['input_type'] == 'image':
                prompt = prompt + '<image>'

        messages = {
            "messages": [
                {"content": prompt, "role": "user"},
                {"content": response, "role": "assistant"}
            ],
            "images": imgs
        }


        chose_results_flag.add(origin_result["question"] + origin_result["answer"])
        sft_results.append(messages)
        # rl数据集区域
        if origin_result['question'] not in rl_flag or origin_result['answer'] == 'Not answerable':
            if rm_score < 28:
                continue
            if len(origin_result['evidence_pages']) <= 1 and origin_result['answer'] != 'Not answerable':
                answerable_num_in_1_rl += 1
                if answerable_num_in_1_rl % 2 != 0:
                    continue
            rl_flag.append(origin_result['question'])
            if len(origin_result['evidence_pages']) <= 1 and origin_result['answer'] == 'Not answerable':
                not_answerable_num_in_1_rl += 1
                if not_answerable_num_in_1_rl % 4 != 0:
                    continue
            if origin_result['answer'] == 'Not answerable':
                not_answerable_num_in_rl += 1
            if len(origin_result['evidence_pages']) <= 1:
                single_reason_num_in_rl += 1
            else:
                multi_reason_num_in_rl += 1
            rl_results.append({
                "messages": [
                    {"content": prompt, "role": "user"},
                    {"content": response, "role": "assistant"}
                ],
                "images": imgs
            })

    for origin_result in tqdm(origin_results):
        if (origin_result["question"] + origin_result["answer"] in chose_results_flag
                or origin_result["answer"] == "Not answerable"):
            continue
        else:
            cleaned_results.append(origin_result)

    rl_results_copy = copy.deepcopy(rl_results)
    not_answerable_data = swap_multimodal_data(rl_results_copy)
    rl_results.extend(not_answerable_data[:2000])
    print(error_0, error_1, error_2, error_3, error_4)
    print(f"原始数据项总数: {len(origin_results)}")
    print(f"清理并筛选后的sft数据项总数: {len(sft_results)}")
    print(f"清理并筛选后的sft数据项中的单跳问题总数: {single_reason_num_in_sft}")
    print(f"清理并筛选后的sft数据项中的多跳问题总数: {multi_reason_num_in_sft}")
    print(f"清理并筛选后的rl数据项总数: {len(rl_results)}")
    print(f"清理并筛选后的rl数据项中的单跳问题总数: {single_reason_num_in_rl}")
    print(f"清理并筛选后的rl数据项中的多跳问题总数: {multi_reason_num_in_rl}")
    print(f"清理并筛选后的rl数据项中的 not answerable 总数: {not_answerable_num_in_rl}")
    print(f"构造的rl空数据总数: {len(not_answerable_data)}")
    print(f"可补充的被清理的数据总数: {len(cleaned_results)}")
    # if "mrag-train" in args.save_file:
    #     random.shuffle(sft_results)
    #     write_json(sft_results, f'{args.save_file}/sft_dataset.json')
    #     write_json(sft_results, '/data/npy/output/datasets/mrag-train/sft_dataset.json')
    #     random.shuffle(rl_results)
    #     write_json(rl_results, f'{args.save_file}/rl_dataset.json')
    #     write_json(rl_results, f'/data/npy/output/datasets/mrag-train/rl_dataset.json')
    #
    #     write_json(cleaned_results, f'{args.save_file}/add_dataset.json')
    # elif "mrag-test" in args.save_file:
    #     random.shuffle(sft_results)
    #     write_json(sft_results, f'{args.save_file}/mrag_dataset.json')
    #     write_json(sft_results, '/data/npy/output/datasets/mrag-test/mrag_dataset.json')

    if prompt_lengths:
        plot_token_distribution(prompt_lengths, f'{args.save_file}/prompt_distribution.png')

    plot_rm_scores(all_scores)
    with open(f'{args.save_file}/scores_data.jsonl', 'w', encoding='utf-8') as f:
        for score, prompt, answer in zip(all_scores, all_scores_prompt, all_scores_answer):
            # 将每一对数据存为一个字典对象
            data = {
                "score": score,
                "prompt": prompt,
                "response": answer
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
