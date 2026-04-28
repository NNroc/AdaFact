import os
import sys
import logging
import warnings
from dataclasses import dataclass

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/data/npy/models/cache'
sys.path.append("../../")
warnings.filterwarnings("ignore", category=UserWarning)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

logger = logging.getLogger("multimodalrag")


@dataclass
class GlobalConfig:
    top_k: int = 30
    chunk_max_token_size: int = 400
    chunk_overlap_token_size: int = 20
    embedding_batch_num: int = 32
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_local_context: int = 6000
    tiktoken_model_name: str = 'gpt-4o'

    embeddings_sentences_cosine_threshold: float = 0.3
    embeddings_entities_cosine_threshold: float = 0.6

    work_dir: str = "/home/npy"
    mineru_dir: str = ""
    API_KEY: str = ""
    MODEL: str = "Qwen2.5-7B-Instruct"
    URL: str = ""
    MM_API_KEY: str = ""
    MM_MODEL: str = "Qwen2.5-VL-7B-Instruct"
    MM_URL: str = ""
    TEXT_ENCODER_MODEL: str = 'Qwen/Qwen3-Embedding-4B'
    MM_ENCODER_MODEL: str = 'colqwen2.5-v0.2'
    prompt_method: str = 'test'
    temperature: float = 0.0

    # def update_global_config(self, args):
    #     self.work_dir = getattr(args, 'work_dir', self.work_dir)
    #     self.top_k = getattr(args, 'top_k', self.top_k)
    #     self.MODEL = getattr(args, 'MODEL', self.MODEL)
    #     self.URL = getattr(args, 'URL', self.URL)
    #     self.MM_MODEL = getattr(args, 'MM_MODEL', self.MM_MODEL)
    #     self.MM_URL = getattr(args, 'MM_URL', self.MM_URL)
    #     self.TEXT_ENCODER_MODEL = getattr(args, 'TEXT_ENCODER_MODEL', self.TEXT_ENCODER_MODEL)
    #     self.MM_ENCODER_MODEL = getattr(args, 'MM_ENCODER_MODEL', self.MM_ENCODER_MODEL)
    #     self.prompt_method = getattr(args, 'prompt_method', self.prompt_method)
    #     self.temperature = getattr(args, 'temperature', self.temperature)

    def update_config(self, args):
        if args is None:
            return
        args_dict = vars(args) if not isinstance(args, dict) else args
        for key, value in args_dict.items():
            # 只有当 key 是本类已有的属性时才更新
            if hasattr(self, key):
                setattr(self, key, value)


global_config = GlobalConfig()
