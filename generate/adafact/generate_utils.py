import copy
import random

from tqdm.contrib.concurrent import thread_map

from src.operations.llm_operation import information_process
from generate.adafact.prompt import PROMPTS
from llm.vllm import multimodel_if_cache
from utils.file_utils import write_json
from utils.storage import JsonKVStorage

API_PORTS = ["http://localhost:8011/v1", "http://localhost:8012/v1"]
# API_PORTS = ["http://localhost:8012/v1"]

def run_generate_process(use_combinations, save_file_path):
    response_cache_1 = JsonKVStorage(
        namespace=f'generate_cache_1',
        global_config={'working_dir': save_file_path}
    )
    response_cache_2 = JsonKVStorage(
        namespace=f'generate_cache_2',
        global_config={'working_dir': save_file_path}
    )
    response_cache_3 = JsonKVStorage(
        namespace=f'generate_cache_3',
        global_config={'working_dir': save_file_path}
    )

    def generate_process_1(comb):
        try:
            prompt = PROMPTS["generate_step_1"].format(query=comb['question'])
            response, history = multimodel_if_cache(user_prompt=prompt,
                                                    max_tokens=8192,
                                                    temperature=0.1,
                                                    customize_api_url=random.choice(API_PORTS),
                                                    hashing_kv=response_cache_1
                                                    )
        except Exception as e:
            print(f"[ERROR] generate_process 1 failed: {e}")
            return None
        history.append({"role": "assistant", "content": response})
        comb['history'] = history
        return comb

    def generate_process_2(comb):
        if comb is None or len(comb['history']) < 2:
            return None
        try:
            prompt = PROMPTS["generate_step_2"].format(query=comb['question'])
            comb_history = copy.deepcopy(comb['history'])
            comb_history[0]['content'] = comb_history[0]['content'][0]['text'].split('\n##########')[0].strip()
            response, history = multimodel_if_cache(user_prompt=prompt,
                                                    extra_information=information_process(
                                                        comb['use_retrieve_information']),
                                                    history_messages=comb_history,
                                                    max_tokens=8192,
                                                    temperature=0.1,
                                                    customize_api_url=random.choice(API_PORTS),
                                                    hashing_kv=response_cache_2
                                                    )
        except Exception as e:
            print(f"[ERROR] generate_process 2 failed: {e}")
            return None
        history.append({"role": "assistant", "content": response})
        comb['history'] = history
        return comb

    def generate_process_3(comb):
        try:
            prompt = PROMPTS["generate_step_3"].format(query=comb['question'], answer=comb['answer'])
            comb_history = copy.deepcopy(comb['history'])
            comb_history[2]['content'][0]['text'] = comb_history[2]['content'][0]['text'].split('\n##########')[0].strip()
            response, history = multimodel_if_cache(user_prompt=prompt,
                                                    history_messages=comb_history,
                                                    max_tokens=8192,
                                                    temperature=0.1,
                                                    customize_api_url=random.choice(API_PORTS),
                                                    hashing_kv=response_cache_3
                                                    )
        except Exception as e:
            print(f"[ERROR] generate_process 3 failed: {e}")
            return None
        history.append({"role": "assistant", "content": response})
        comb['history'] = history
        return comb

    generate_results = []
    try:
        generate_results_1 = thread_map(generate_process_1, use_combinations, desc="Generating 1...", max_workers=32)
        generate_results_2 = thread_map(generate_process_2, generate_results_1, desc="Generating 2...", max_workers=32)
        generate_results_3 = thread_map(generate_process_3, generate_results_2, desc="Generating 3...", max_workers=32)
        generate_results = generate_results_3
    except Exception as e:
        print(f"[ERROR] generate_process failed: {e}")
    finally:
        response_cache_1.index_done_callback()
        response_cache_2.index_done_callback()
        response_cache_3.index_done_callback()
    cleaned_generate_results = []
    for r in generate_results:
        if r is None:
            continue
        history_messages = []
        for history in r['history']:
            if history['role'] == 'assistant':
                history_messages.append(history['content'])
        r['history'] = history_messages
        cleaned_generate_results.append(r)

    write_json(cleaned_generate_results, f'{save_file_path}/origin_dataset.json')
