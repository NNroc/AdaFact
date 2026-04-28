import logging
import re
import html
import numbers
import ast
import sys
import numpy as np
from hashlib import md5
from typing import Any


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input
    # 移除 HTML 转义字符
    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    # 移除控制字符和其他不需要的字符
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# 计算参数的哈希值
def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


# 计算md5哈希值
def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


# 判断是否为浮点数
def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def remove_null_chars(string: str) -> str:
    INVALID_XML_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')
    return INVALID_XML_RE.sub("", string)


def parse_value(value):
    """
    解析字符串值，将其转换为适当的 Python 类型（字典、数字、字符串等）。
    """
    try:
        # 使用 ast.literal_eval 解析字典或其他复杂结构
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # 如果无法解析，则返回原始字符串
        return value


# 将字符串括在双引号中
def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


# 将多维列表转换为CSV格式
def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


def convert_numpy_to_python(obj):
    """递归转换NumPy数组为Python列表"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # 处理NumPy数组
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
