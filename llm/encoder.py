import os
import torch
from typing import Counter
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from colpali_engine import ColPali, ColPaliProcessor, ColQwen2_5, ColQwen2_5_Processor
from transformers.utils import is_flash_attn_2_available


def information_encode(encoder, text=None, images=None, batch_size=32):
    assert (isinstance(encoder, SentenceTransformer) or isinstance(encoder, tuple)), \
        f"information_encode encoder type error: {type(encoder)}"
    if isinstance(encoder, SentenceTransformer):
        return encoder.encode(text)
    else:
        model, processor = encoder
        all_embeddings = torch.Tensor().to(model.device)

        # 调整所有图片到众数尺寸
        sizes = [img.size for img in images]
        mode_size = Counter(sizes).most_common(1)[0][0]
        resized_images = []
        for img in images:
            if img.size == mode_size:
                resized_images.append(img)  # 保持原图
            else:
                resized_images.append(img.resize(mode_size))  # 调整尺寸
        images = resized_images

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_information = processor.process_images(images=batch_images).to(model.device)
            with torch.no_grad():
                batch_embeddings = model(**batch_information)
            all_embeddings = torch.cat((all_embeddings, batch_embeddings), dim=0)
        return all_embeddings


@lru_cache(maxsize=2)
def init_model(model_name, models_root='/data/npy/models/'):
    if 'all-MiniLM-L6-v2' in model_name:
        return SentenceTransformer(f'{models_root}sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    elif 'Qwen3-Embedding-0.6B' in model_name:
        return SentenceTransformer(f'{models_root}Qwen/Qwen3-Embedding-0.6B', device='cuda')
    elif 'Qwen3-Embedding-4B' in model_name:
        return SentenceTransformer(f'{models_root}Qwen/Qwen3-Embedding-4B', device='cuda')
    elif 'Qwen3-Embedding-8B' in model_name:
        return SentenceTransformer(f'{models_root}Qwen/Qwen3-Embedding-8B', device='cuda')
    elif 'NV-Embed-v2' in model_name:
        return SentenceTransformer(f'{models_root}nvidia/NV-Embed-v2', device='cuda', trust_remote_code=True)
    elif 'colpali-v1.3' in model_name:
        model = ColPali.from_pretrained(f'{models_root}vidore/colpali-v1.3-merged',
                                        dtype=torch.bfloat16,
                                        device_map='cuda').eval()
        processor = ColPaliProcessor.from_pretrained(f'{models_root}/vidore/colpali-v1.3-merged')
        return model, processor
    elif 'colqwen2.5-v0.2' in model_name:
        model = ColQwen2_5.from_pretrained(f'{models_root}vidore/colqwen2.5-v0.2',
                                           dtype=torch.bfloat16,
                                           device_map='cuda',
                                           attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
                                           ).eval()
        processor = ColQwen2_5_Processor.from_pretrained(f'{models_root}/vidore/colqwen2.5-v0.2')
        return model, processor
    elif 'colqwen2.5-3b-multilingual-v1.0' in model_name:
        model = ColQwen2_5.from_pretrained(f'{models_root}tsystems/colqwen2.5-3b-multilingual-v1.0',
                                           dtype=torch.bfloat16,
                                           device_map='cuda').eval()
        processor = ColQwen2_5_Processor.from_pretrained(f'{models_root}tsystems/colqwen2.5-3b-multilingual-v1.0')
        return model, processor
    else:
        raise Exception(f"unknown model name: {model_name}")


def get_model_dim(model_name):
    if 'all-MiniLM-L6-v2' in model_name:
        return 384
    elif 'Qwen3-Embedding-0.6B' in model_name:
        return 1024
    elif 'Qwen3-Embedding-4B' in model_name:
        return 2560
    elif 'Qwen3-Embedding-8B' in model_name:
        return 4096
    elif 'NV-Embed-v2' in model_name:
        return 1024
    elif 'colpali-v1.3' in model_name:
        return 2048
    elif 'colqwen2.5-3b-multilingual-v1.0' in model_name:
        return 3072
    else:
        raise Exception(f"unknown model name: {model_name}")
