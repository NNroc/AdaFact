import tiktoken
import numpy as np
from dataclasses import dataclass

Float = np.float32
ENCODER = None


@dataclass
class EmbeddingFunc:
    """
    定义一个用于嵌入的函数类。

    参数:
    - embedding_dim: 嵌入向量的维度
    - max_token_size: 最大令牌大小
    - func: 可调用的对象，用于执行嵌入操作
    """
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        调用嵌入函数并返回结果。

        参数:
        - *args: 位置参数
        - **kwargs: 关键字参数

        返回:
        - np.ndarray: 嵌入结果
        """
        return await self.func(*args, **kwargs)


# 使用指定的模型encode字符串，并返回tokens数量
def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


# 使用指定的模型decode字符串，并返回tokens数量
def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    """
    根据token大小截断列表数据。

    该函数的目的是确保列表中数据的总token数不超过指定的最大token大小。
    当数据的总token数超过最大允许大小时，函数将返回截断后的列表。

    参数:
    - list_data: list, 需要截断的列表，其中每个元素为一个数据项。
    - key: callable, 用于从列表数据项中提取用于计算token大小的字符串的函数。
    - max_token_size: int, 允许的最大token大小，用于决定列表数据的截断点。

    返回:
    - 截断后的列表。如果max_token_size小于等于0，返回空列表。

    注意:
    - 该函数使用tiktoken对字符串进行编码并计算token数量，请确保在使用前已安装tiktoken库。
    - 截断操作基于累计token数量首次超过max_token_size发生的索引位置。
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data
