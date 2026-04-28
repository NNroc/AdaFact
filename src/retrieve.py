import copy
import os
import argparse
import logging
from config.base_config import global_config
from src.model.mrag import Model
from src.shared_parser import get_shared_parser
from utils.base_utils import compute_args_hash
from utils.decorator_utils import parallel_processor
from utils.file_utils import load_json, write_json


@parallel_processor(mode="thread", max_workers=1, desc='Retrieve')
def retrieve(sample):
    doc_id = sample['doc_id'].replace('.pdf', '')
    working_dir = f'{args.databases_save_dir}/{args.dataset}{args.prefix}/{doc_id}'

    if all_retrieval_results is None or sample['id'] not in all_retrieval_results:
        retrieve_model = Model(working_dir=working_dir, dataset=args.dataset)
        retrieval_results, retrieve_page_list, retrieve_page_scores \
            = retrieve_model.retrieve_document(query=sample['question'],
                                               method=args.method,
                                               top_k=args.top_k,
                                               question_decomposition_cache=None)

        if retrieval_results is not None:
            sample['pages_ranking'] = str(retrieve_page_list)
            sample['pages_scores'] = str(retrieve_page_scores)
            all_retrieval_results[sample['id']] = retrieval_results


if __name__ == '__main__':
    shared_parser = get_shared_parser()
    parser = argparse.ArgumentParser(parents=[shared_parser])
    parser.add_argument('--entity_top_k', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=60)
    parser.add_argument('--method', type=str, default='colqwen',
                        choices=['base', 'colpali', 'colqwen', 'all'])
    args = parser.parse_args()
    if args.debug:
        global_config.logger.setLevel(logging.DEBUG)
    global_config.update_config(args)

    # load dataset's qa
    samples = load_json(f'{args.dataset_dir}/samples_{args.dataset}.json')
    retrieve_working_dir = f'{args.databases_save_dir}/{global_config.MM_MODEL}/'
    retrieve_results_eval_file = f'{args.results_save_dir}/retrieve_results/{args.dataset}_{args.method}_retrieval_results_eval.json'
    retrieve_results_file = f'{args.results_save_dir}/retrieve_results/{args.dataset}_{args.method}_retrieval_results.json'

    all_retrieval_results_eval = load_json(retrieve_results_eval_file)
    all_retrieval_results = load_json(retrieve_results_file)
    if all_retrieval_results_eval is None:
        os.makedirs(retrieve_working_dir, exist_ok=True)
        all_retrieval_results_eval = copy.deepcopy(samples)
        all_retrieval_results = {}

    retrieve(all_retrieval_results_eval)

    write_json(all_retrieval_results_eval, retrieve_results_eval_file)
    write_json(all_retrieval_results, retrieve_results_file)
