from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import tiktoken


def count_tokens(text: str, tiktoken_model="gpt-4o"):
    """
    计算字符串的 tokens 数量
    """
    encoder = tiktoken.encoding_for_model(tiktoken_model)
    encoder.encode(text)
    return len(encoder.encode(text))


def count_text_tokens(text: str):
    """
    使用 Qwen2.5-VL 分词器计算 tokens 数量
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/npy/models/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True
    )
    return len(tokenizer.encode(text))


def count_image_tokens(image_path: str):
    processor = AutoProcessor.from_pretrained(
        "/data/npy/models/Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True
    )
    image = Image.open(image_path)

    # 直接调用 image_processor 绕过文本逻辑
    inputs = processor.image_processor(images=image, return_tensors="pt")

    grid_thw = inputs["image_grid_thw"]
    num_tokens = grid_thw[:, 1] * grid_thw[:, 2]

    return int(num_tokens)
