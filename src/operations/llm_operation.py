import ast
import json
import re

from PIL import Image

from llm.vllm import multimodel_if_cache
from src.prompt.prompt import PROMPTS
from config.base_config import logger, global_config
from utils.base_utils import split_string_by_multi_markers, remove_null_chars
from utils.file_utils import load_base64

context_base = dict(
    language=PROMPTS["DEFAULT_LANGUAGE"],
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
)


def process_information(information_list, image_min_pixels=512 * 28 * 28, image_max_pixels=5120 * 28 * 28):
    messages = []
    messages_content = []
    for input_information in information_list:
        if input_information['type'] == "text":
            messages_content.append({
                "type": "text",
                "text": input_information["content"]
            })
        elif input_information["type"] == "image":
            if isinstance(input_information["content"], str):
                input_information["content"] = Image.open(input_information["content"]).convert('RGB')
            messages_content.append({
                "type": "image",
                "image": input_information["content"],
                "max_pixels": image_max_pixels,
                "min_pixels": image_min_pixels,
            })
        else:
            raise NotImplementedError
    messages.append({"role": "user", "content": messages_content})
    return messages


def information_process(information_list):
    message_content = []
    for input_information in information_list:
        if 'input_content' in input_information and input_information['input_content']:
            message_content.append({
                'type': 'text',
                'text': input_information['input_content'] + '\n'
            })
        if input_information['input_type'] == 'image':
            message_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{load_base64(input_information['input_path'])}"}
            })
    return message_content


def information_process_test(information_list):
    message_content = []
    for input_information in information_list:
        content_text = ''
        if 'input_id' in input_information and input_information['input_id']:
            content_text = content_text + input_information['input_id'] + '\n'
        if 'input_page_idx' in input_information and input_information['input_page_idx']:
            content_text = content_text + 'Page Number: ' + str(input_information['input_page_idx']) + '\n'
        if 'input_content' in input_information and input_information['input_content']:
            content_text = content_text + input_information['input_content'] + '\n'
        if content_text:
            message_content.append({
                'type': 'text',
                'text': content_text
            })
        if input_information['input_type'] == 'image':
            message_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{load_base64(input_information['input_path'])}"}
            })
    return message_content


def image_summary_by_mllm(img_base, hashing_kv=None):
    content, maybe_description, maybe_caption, maybe_type, maybe_entities, maybe_relationships = '', '', '', '', set(), set()

    examples = PROMPTS["image_description_examples"].format(**context_base)
    prompt = PROMPTS["image_description"].format(**context_base,
                                                 examples=examples)
    retry_count, max_retries = 1, 3
    prompt_use = prompt
    while retry_count <= max_retries:
        # use mllm summary and extraction key information
        results, _ = multimodel_if_cache(user_prompt=prompt_use,
                                         information_list=img_base,
                                         max_tokens=36000,
                                         temperature=0.1,
                                         customize_api_url=global_config.MM_URL,
                                         hashing_kv=hashing_kv)
        caption_match = re.search(r'"caption"\s*:\s*"([^"]*)"', results)
        type_match = re.search(r'"type"\s*:\s*"([^"]*)"', results)
        desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', results)

        maybe_caption = caption_match.group(1) if caption_match else ""
        maybe_type = type_match.group(1) if type_match else ""
        maybe_description = desc_match.group(1) if desc_match else ""
        if len(maybe_caption) > 0 and len(maybe_description) > 0:
            retry_count += 3
        if maybe_description == "":
            prompt_use = f'Output description format error, retry {retry_count}-th time.\n\nLast response:\n' + results + '\n\n' + prompt
        retry_count += 1

    if not maybe_description.strip() and retry_count == max_retries:
        logger.info('image_summary_by_mllm: Maximum number of retries reached, using the last result.')
    if maybe_description == "":
        cleaned_results = re.sub(r'"caption"\s*:\s*"[^"]*"\s*,?\s*', '', results)
        desc_match = re.search(r'"description"\s*:\s*(.*)', cleaned_results, flags=re.S)
        maybe_description = desc_match.group(1) if desc_match else ''
        if '<|endofprompt|>' in maybe_description:
            maybe_description = maybe_description.replace('<|endofprompt|>', '')
        maybe_description = maybe_description.strip().lstrip('{[" \n\t').rstrip('"}] \n\t')

        maybe_description = re.sub(r'(\b[\w,.:;!?()\'"\\$]+[\s]*)\1+$', r'\1', maybe_description, flags=re.S)
        if maybe_description == '':
            print('description is empty!')

        if 'chart' in maybe_type.lower():
            maybe_type = 'Chart'
        elif 'table' in maybe_type.lower():
            maybe_type = 'Table'
        else:
            maybe_type = 'Figure'

    maybe_description = remove_null_chars(maybe_description)
    maybe_type = remove_null_chars(maybe_type)
    maybe_caption = remove_null_chars(maybe_caption)
    return content, maybe_caption, maybe_type, maybe_description, list(maybe_entities), list(maybe_relationships)


def query_answer_by_mllm(query, extra_information, answer_prompt='question_answer_baseline', hashing_kv=None):
    prompt = PROMPTS[answer_prompt].format(**context_base, query=query)
    results, _ = multimodel_if_cache(user_prompt=prompt,
                                     extra_information=extra_information,
                                     max_tokens=4096,
                                     temperature=global_config.temperature,
                                     customize_api_url=global_config.MM_URL,
                                     hashing_kv=hashing_kv)
    return results
