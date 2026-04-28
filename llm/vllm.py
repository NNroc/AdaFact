import torch

from functools import lru_cache
from openai import OpenAI, RateLimitError, APIConnectionError, Timeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.base_utils import compute_args_hash
from utils.storage import BaseKVStorage
from config.base_config import global_config


# @retry(
#     stop=stop_after_attempt(5),
#     wait=wait_exponential(multiplier=2, min=8, max=20),
#     retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
# )
# def model_if_cache(prompt, max_tokens=32768, system_prompt=None, temperature=0.1,
#                    history_messages=None, customize_api_url=None, customize_api_key='API_KEY',
#                    **kwargs) -> str:
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#
#     hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
#     if history_messages:
#         messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})
#     # 如果缓存对象存在，计算当前请求的哈希值，尝试从缓存中获取结果
#     if hashing_kv is not None:
#         args_hash = compute_args_hash(global_config.MODEL, messages)
#         if_cache_return = hashing_kv.get_by_id(args_hash)
#         if if_cache_return is not None:
#             return if_cache_return["return"]
#
#     if customize_api_url:
#         openai_client = OpenAI(api_key=customize_api_key, base_url=customize_api_url)
#     else:
#         openai_client = OpenAI(api_key=global_config.API_KEY, base_url=global_config.URL)
#     response = openai_client.chat.completions.create(
#         model=global_config.MODEL, messages=messages, max_tokens=max_tokens, frequency_penalty=0.5,
#         temperature=temperature, **kwargs
#     )
#     result = response.choices[0].message.content
#     completion_tokens = response.usage.completion_tokens
#     prompt_tokens = response.usage.prompt_tokens
#
#     # 如果有缓存对象，将响应结果存入缓存
#     if hashing_kv is not None:
#         hashing_kv.upsert(
#             {args_hash: {"return": result, "model": global_config.MM_MODEL,
#                          "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens}}
#         )
#         hashing_kv.index_done_callback()
#     return result


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=8, max=20),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
def multimodel_if_cache(
        user_prompt, extra_information=None, system_prompt=None, max_tokens=32768, temperature=0.7,
        history_messages=None, customize_api_url=None, customize_api_key='', **kwargs
):
    messages, message_content = [], []
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if system_prompt:
        messages.append({"role": "system", "content": [{
            "type": "text",
            "text": system_prompt
        }]})
    if history_messages:
        messages.extend(history_messages)
    message_content.append({"type": "text", "text": user_prompt})
    if extra_information:
        message_content.extend(extra_information)
    messages.append({'role': 'user', 'content': message_content})

    # 如果缓存对象存在，计算当前请求的哈希值，尝试从缓存中获取结果
    if hashing_kv is not None:
        args_hash = compute_args_hash(global_config.MM_MODEL, messages)
        if_cache_return = hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"], messages

    if customize_api_url:
        openai_client = OpenAI(api_key=customize_api_key, base_url=customize_api_url)
        response = openai_client.chat.completions.create(
            model=global_config.MM_MODEL, messages=messages, max_tokens=max_tokens, frequency_penalty=0.5,
            temperature=temperature, **kwargs
        )
        result = response.choices[0].message.content
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
    else:
        from vllm import SamplingParams, LLM
        llm = load_vllm(global_config.MM_MODEL)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=kwargs.get("frequency_penalty", 0.5)
        )

        response = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        result = response[0].outputs[0].text
        completion_tokens = len(response[0].outputs[0].token_ids)
        prompt_tokens = len(response[0].prompt_token_ids)

    # 如果有缓存对象，将响应结果存入缓存
    if hashing_kv is not None:
        hashing_kv.upsert(
            {args_hash: {"return": result, "model": global_config.MM_MODEL,
                         "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens}}
        )
    return result, messages


def invoke_openai_api(model_name="Qwen2.5-32B-Instruct", content='', temperature=0.0):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    if global_config.URL:
        client = OpenAI(api_key=global_config.API_KEY, base_url=global_config.URL)
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=256,
            top_p=0.1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content.strip()
    else:
        from vllm import SamplingParams, LLM
        llm = load_vllm(model_name)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=256,
            frequency_penalty=0
        )
        response = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        return response[0].outputs[0].text.strip()


@lru_cache(maxsize=1)
def load_vllm(model_name: str):
    from vllm import LLM
    # 使用vllm 只支持单线程
    basepath = f"{global_config.work_dir}/models"
    device_properties = torch.cuda.get_device_properties(0)
    total_memory_gb = device_properties.total_memory / (1024 ** 3)
    tp_size = 2 if total_memory_gb < 80 and any(size in model_name for size in ['30B', '32B']) else 1
    if 'test' in model_name:
        model_path = f'{global_config.work_dir}/output/models/{model_name}'
        llm_instance = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            mm_processor_cache_gb=0,
            trust_remote_code=True,
        )
    elif 'Qwen' in model_name and 'Instruct' in model_name:
        model_path = f'{basepath}/Qwen/{model_name}'
        llm_instance = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            mm_processor_cache_gb=0,
            trust_remote_code=True,
        )
    elif 'EVisRAG' in model_name:
        model_path = f'{basepath}/others/{model_name}'
        llm_instance = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            mm_processor_cache_gb=0,
            trust_remote_code=True,
        )
    elif 'Qwen2.5-VL-7B-VRAG' in model_name:
        model_path = f'{basepath}/others/Qwen2.5-VL-7B-VRAG'
        llm_instance = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=32768,
            gpu_memory_utilization=0.8,
            mm_processor_cache_gb=0,
            trust_remote_code=True,
        )
    else:
        raise Exception("Unknown model")
    return llm_instance
