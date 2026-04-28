import datasets
import pandas as pd
import seaborn as sns
from datasets import tqdm
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch
import re
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('/data/npy/models/Qwen/Qwen3-Embedding-0.6B')
model.to(device)

model_name = "/data/npy/models/Skywork/Skywork-Reward-V2-Llama-3.1-8B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def semantic_match(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True, device=device)
    emb2 = model.encode(text2, convert_to_tensor=True, device=device)
    return util.pytorch_cos_sim(emb1, emb2).item()


def get_raw_score(prompt_text, response_text):
    conv = [{"role": "user", "content": prompt_text}, {"role": "assistant", "content": response_text}]
    conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and conv_formatted.startswith(tokenizer.bos_token):
        conv_formatted = conv_formatted[len(tokenizer.bos_token):]
    inputs = tokenizer(conv_formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        score = rm(**inputs).logits[0][0].item()
    return score


def plot_score_distribution(scores):
    """绘制分数分布图"""
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # 绘制直方图和核密度估计曲线
    plt.subplot(1, 2, 1)
    sns.histplot(scores, kde=True, color="skyblue", bins=30)
    plt.title('Skywork Score Distribution (Gold Responses)')
    plt.xlabel('Raw Score (Logits)')
    plt.ylabel('Frequency')

    # 绘制箱线图看离群点
    plt.subplot(1, 2, 2)
    sns.boxplot(y=scores, color="lightsalmon")
    plt.title('Score Boxplot')
    plt.ylabel('Raw Score')

    plt.tight_layout()
    plt.savefig(f"gold_score_distribution_{model_name}.png")
    print(f"图表已保存至: gold_score_distribution_{model_name}.png")
    plt.show()


def get_gold_scores(dataset):
    scores = []
    print(f"开始计算 {len(dataset)} 条数据的标答得分...")
    for i in tqdm(range(len(dataset))):
        example = dataset[i]
        prompt = example['messages'][0]['content']
        gold_response = example['messages'][1]['content']
        match = re.search(r"<answer>(.*?)</answer>", gold_response, re.S)
        answer = match.group(1).strip()
        prompt = prompt + f"\nThe golden answer is: {answer}"
        try:
            score = get_raw_score(prompt, gold_response)
            if score < 15.0:
                print(score)
                print(gold_response)
            scores.append(score)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue
    return scores


if __name__ == "__main__":
    local_dataset_path = "/data/npy/output/datasets/mrag-train"
    json_file_path = os.path.join(local_dataset_path, 'rl_dataset.json')
    dataset = datasets.load_dataset("json", data_files=json_file_path)
    total_size = len(dataset["train"])
    num = total_size
    all_dataset = dataset["train"].select(range(0, num))

    gold_scores = get_gold_scores(all_dataset)

    # 输出基础统计信息
    df_scores = pd.DataFrame(gold_scores, columns=['score'])
    print("\n--- 分数统计摘要 ---")
    print(df_scores.describe())

    # 绘图
    plot_score_distribution(gold_scores)
