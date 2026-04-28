import os
import argparse
import logging
import copy
import traceback

import config.base_config as base_config
from src.model.mrag import Model
from src.shared_parser import get_shared_parser
from utils.decorator_utils import parallel_processor
from utils.file_utils import load_json, write_json
from utils.storage import JsonKVStorage


@parallel_processor(mode="thread", max_workers=32, desc='Answer')
def answer(sample):
    doc_id = sample['doc_id'].replace('.pdf', '')
    working_dir = f'{args.databases_save_dir}/{args.dataset}{args.prefix}/{doc_id}'
    if 'raw_response' not in sample:
        answer_model = Model(working_dir=working_dir, dataset=args.dataset)
        answer_results = answer_model.answer(sample=sample,
                                             retrieve_results=all_retrieve_results[sample['id']],
                                             top_k=args.top_k,
                                             response_cache=answer_cache)
        sample['raw_response'] = answer_results


if __name__ == '__main__':
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(parents=[shared_parser])
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--retrieve_method', type=str, default='colqwen', choices=['base', 'colpali', 'colqwen', 'all'])
    parser.add_argument('--prompt_method', type=str, default='question_answer_test')
    args = parser.parse_args()
    if args.debug:
        base_config.logger.setLevel(logging.DEBUG)
    base_config.global_config.update_config(args)

    # load dataset's qa
    retrieve_results_file = f'{args.results_save_dir}/retrieve_results/{args.dataset}_{args.retrieve_method}_retrieval_results.json'
    retrieve_results_eval_file = f'{args.results_save_dir}/retrieve_results/{args.dataset}_{args.retrieve_method}_retrieval_results_eval.json'
    answer_working_dir = f'{args.results_save_dir}/answer_results/{base_config.global_config.MM_MODEL}/'
    answer_results_file = f'{args.results_save_dir}/answer_results/{base_config.global_config.MM_MODEL}/{args.dataset}_{args.retrieve_method}_{args.top_k}_{args.prompt_method}_results.json'

    answer_cache = JsonKVStorage(
        namespace=f'{args.dataset}_{args.retrieve_method}_{args.top_k}_{args.prompt_method}_answer_cache',
        global_config={'working_dir': answer_working_dir}
    )
    print(answer_cache)
    all_retrieve_results = load_json(retrieve_results_file)
    all_retrieve_eval_results = load_json(retrieve_results_eval_file)
    all_answer_results = load_json(answer_results_file)
    if all_retrieve_results is None:
        raise RuntimeError('retrieve_results cannot be None')
    if all_answer_results is None:
        os.makedirs(answer_working_dir, exist_ok=True)
        all_answer_results = copy.deepcopy(all_retrieve_eval_results)

    try:
        answer(all_answer_results)
    except Exception as e:
        print(traceback.format_exc())  # 这行会打印出完整的报错堆栈
    finally:
        answer_cache.index_done_callback()
        write_json(all_answer_results, answer_results_file)
