import os
import json
import argparse
import pandas as pd
from graphrag.query.input import retrieval

from config.base_config import global_config
from evaluate.eval_qa_llm import extract_answer, extract_score, eval_one_sample, eval_samples, \
    show_fine_grained_results
from utils.decorator_utils import parallel_processor
from utils.token_utils import count_tokens


def process_response(raw_response: str):
    pred_answer = ""
    if raw_response.find("<think>") != -1:
        pred_answer = pred_answer + raw_response.split("<think>")[1].split('</think>')[0] + '\n'
    if raw_response.find("<reason>") != -1:
        pred_answer = pred_answer + raw_response.split("<reason>")[1].split('</reason>')[0] + '\n'
    if raw_response.find("<answer>") != -1:
        pred_answer = pred_answer + raw_response.split("<answer>")[1].split('</answer>')[0] + '\n'
    if pred_answer == "":
        pred_answer = raw_response
    return pred_answer


@parallel_processor(mode="thread", max_workers=64, desc='Scoring Samples')
def process_sample(sample):
    if 'score' in sample:
        return sample
    if args.dataset in ["MMLong", "LongDocURL"]:  # for MMLong and LongDocURL, extract answer
        if "pred_ans" not in sample:
            raw_response = sample.get("raw_response", '')
            extracted_ans = extract_answer(question=sample["question"],
                                           output=process_response(raw_response),
                                           extractor_prompt=extractor_prompt,
                                           model_name=args.MODEL)
            try:
                pred_ans = extracted_ans.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                sample["pred_ans"] = pred_ans
            except:
                sample["pred_ans"] = "Failed to extract"

        em_score, acc_score = eval_one_sample(gt=sample["answer"],
                                              pred=sample["pred_ans"],
                                              answer_type=sample["answer_format"])
        sample["score"] = {"EM": em_score, "Acc": acc_score, "tokens": count_tokens(sample.get("raw_response", ''))}

    elif args.dataset in ["PaperTab", "FetaTab", "docvqa"]:  # for PaperTab and FetaTab, assign score
        if "score" not in sample:
            raw_response = sample.get("raw_response", '')
            generated_score = extract_score(question=sample["question"],
                                            output=process_response(raw_response),
                                            ground_truth=sample["answer"],
                                            prompt=scoring_prompt,
                                            model_name=args.MODEL)
            score = generated_score.get('binary_correctness', 0)

            sample["score"] = {"BinaryCorrectness": score, "tokens": count_tokens(sample.get("raw_response", ''))}
    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="answer_results")
    parser.add_argument("--MODEL", type=str, required=True)
    parser.add_argument("--URL", type=str, default='http://localhost:8011/v1')
    args = parser.parse_args()

    global_config.update_config(args)

    cur_pred_path = args.filepath
    print(os.getcwd())
    with open("./evaluate/prompt_for_answer_extraction.md", 'r') as f:
        extractor_prompt = f.read()
    with open(f"./evaluate/prompt_for_scoring.md", 'r') as file:
        scoring_prompt = file.read()

    if not os.path.exists(cur_pred_path):
        raise FileNotFoundError(f"Prediction {cur_pred_path} does not exist.")

    print(cur_pred_path)

    raw_samples, scored_samples = json.load(open(cur_pred_path, 'r')), {}
    rewrite_pred_path = cur_pred_path.replace(".json", f"_scored.json")
    if os.path.exists(rewrite_pred_path):
        scored_samples = json.load(open(rewrite_pred_path, 'r'))

    # Step 1 extract answer / assign scores
    results = process_sample(raw_samples)
    for result in results:
        if result:
            sid, updated_sample = result['id'],result
            scored_samples[sid] = updated_sample

    try:
        assert len(scored_samples) == len(raw_samples)
    except Exception as e:
        print(f"[ERROR] Scored samples {len(scored_samples)} do not match raw samples {len(raw_samples)}. ")
        raise "数量错误"

    # Step 2 Save scored results
    with open(rewrite_pred_path, 'w') as file:
        sorted_items = sorted(
            scored_samples.items(),
            key=lambda x: (x[1].get('doc_id'), x[0])
        )
        scored_samples = dict(sorted_items)
        json.dump(scored_samples, file, indent=4, sort_keys=True)

    # Step 3 Evaluate results
    scored_samples = [sample for k, sample in scored_samples.items()]

    print(f"{args.filepath} results")

    # Step 4 Show fine-grained results
    if args.dataset in ["MMLong", "LongDocURL"]:
        all_results = show_fine_grained_results(scored_samples, args.dataset)
    else:
        score_dict = eval_samples(scored_samples, args.dataset)
        print(f"Evalution-{score_dict}")
        all_results = {"Overall_score": score_dict}

    prefix_path = cur_pred_path.split("/results")[0]
    csv_path = f"{prefix_path}/results/eval_{args.dataset}_results.csv"

    # 保存结果 将all_results追加写入到dataframe
    data_for_df = {}
    for category, metrics in all_results.items():
        clean_cat = category.replace(" Evaluation", "")
        for metric_name, value in metrics.items():
            data_for_df[(clean_cat, metric_name)] = value

    # 使用 MultiIndex.from_tuples 创建双层表头
    new_df = pd.DataFrame([data_for_df])
    new_df.columns = pd.MultiIndex.from_tuples(new_df.columns)

    # 3. 插入模型信息作为索引（或第一列）
    eval_model_info = f"{args.MODEL}"
    qa_model_info = cur_pred_path.split(f"/{args.save_dir}/")[1].split(f"/{args.dataset}_")[0]
    eval_info = cur_pred_path.split("/")[-1].split("_results")[0].split("_")[1:]
    retrieval_model_info = eval_info[0]
    top_k_info = int(eval_info[1])
    prompt_info = "_".join(eval_info[2:])
    new_df.insert(0, ('Model_Config', 'Eval_Model'), eval_model_info)
    new_df.insert(1, ('Model_Config', 'QA_Model'), qa_model_info)
    new_df.insert(2, ('Model_Config', 'retrieval'), retrieval_model_info)
    new_df.insert(3, ('Model_Config', 'prompt'), prompt_info)
    new_df.insert(4, ('Model_Config', 'top_k'), top_k_info)

    # 4. 写入/追加到 CSV
    if os.path.exists(csv_path):
        # 如果文件存在，读取旧数据并合并
        try:
            old_df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            # 检查 new_df 中的信息是否已存在于 old_df 中
            # 注意：MultiIndex 下访问列需要用元组格式
            eval_model_val = new_df[('Model_Config', 'Eval_Model')].iloc[0]
            qa_model_val = new_df[('Model_Config', 'QA_Model')].iloc[0]
            prompt_val = new_df[('Model_Config', 'prompt')].iloc[0]

            # 判断是否存在整行匹配
            is_duplicate = ((old_df[('Model_Config', 'Eval_Model')] == eval_model_val) &
                            (old_df[('Model_Config', 'QA_Model')] == qa_model_val) &
                            (old_df[('Model_Config', 'prompt')] == prompt_val)).any()

            if is_duplicate:
                print(f"检测到重复记录: {eval_model_val} | {qa_model_val} | {prompt_val}，进行覆盖。")

                # 找到重复行的索引
                duplicate_mask = ((old_df[('Model_Config', 'Eval_Model')] == eval_model_val) &
                                  (old_df[('Model_Config', 'QA_Model')] == qa_model_val) &
                                  (old_df[('Model_Config', 'prompt')] == prompt_val))

                # 删除旧数据中的重复行
                old_df_filtered = old_df[~duplicate_mask]

                # 将新数据追加到删除重复行后的数据中
                final_df = pd.concat([old_df_filtered, new_df], ignore_index=True)
            else:
                # 无重复记录，直接追加
                final_df = pd.concat([old_df, new_df], ignore_index=True)
        except Exception:
            # 如果读取失败（可能是格式不匹配），直接覆盖
            final_df = new_df
    else:
        final_df = new_df
    # 指定排序优先级：1. QA_Model -> 2. prompt -> 3. Eval_Model
    sort_columns = [
        ('Model_Config', 'Eval_Model'),
        ('Model_Config', 'retrieval'),
        ('Model_Config', 'top_k'),
        ('Model_Config', 'QA_Model'),
        ('Model_Config', 'prompt'),
    ]
    final_df = final_df.sort_values(by=sort_columns, ascending=True).reset_index(drop=True)
    # --- 1. 保存标准 CSV ---
    final_df.to_csv(csv_path, index=True)

    # --- 2. 保存标准 XLSX (带自动列宽) ---
    xlsx_path = csv_path.replace(".csv", ".xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, index=True, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        # 这里的 i 对应 Excel 的列，包括 MultiIndex 的索引列
        for i, col in enumerate(final_df.reset_index().columns):
            # 针对 MultiIndex，col 是一个 tuple，取最大长度
            col_name_len = 8
            data_len = final_df.reset_index()[col].astype(str).map(len).max()
            column_len = max(data_len, col_name_len)
            worksheet.set_column(i, i, column_len)

    print(f"Evaluation finished!\n\n")
