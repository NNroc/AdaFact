import os
import tiktoken
import torch

from functools import lru_cache
from dataclasses import dataclass, asdict
from typing import Type
from pathlib import Path

from config.base_config import logger, global_config
from llm.encoder import get_model_dim, init_model, information_encode
from src.operations.colqwen_retrieve_operation import DocumentRetriever
from src.operations.llm_operation import query_answer_by_mllm, information_process_test, information_process, \
    image_summary_by_mllm
from utils.base_utils import compute_mdhash_id
from utils.chunking_utils import chunking_by_token_size
from utils.decorator_utils import lazy_property
from utils.storage import BaseKVStorage, JsonKVStorage, BaseVectorStorage, BaseGraphStorage, NanoVectorDBStorage, \
    IGraphStorage


def compare_blocks(a, b):
    """a、b 是两个 unit_id 对应的列表"""
    for x, y in zip(a, b):
        if x[1] != y[1]:
            return -1 if x[1] < y[1] else 1
    return -1 if len(a) < len(b) else 1 if len(a) > len(b) else 0


@lru_cache(maxsize=2)
@dataclass
class Model:
    dataset: str = ''
    working_dir: str = ''
    llm_model_max_token_size: int = 32768
    mllm_model_max_token_size: int = 32768
    embedding_batch_num: int = global_config.embedding_batch_num
    key_value_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_storage_cls: Type[BaseVectorStorage] = NanoVectorDBStorage
    graph_storage_cls: Type[BaseGraphStorage] = IGraphStorage

    @lazy_property
    def mm_encoder(self):
        return init_model(global_config.MM_ENCODER_MODEL)

    def __post_init__(self):
        self.text_encoder_name = global_config.TEXT_ENCODER_MODEL
        self.text_embedding_dim = get_model_dim(global_config.TEXT_ENCODER_MODEL)
        self.mm_encoder_name = global_config.MM_ENCODER_MODEL

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)
        # 初始化存储类实例，用于存储原始文本块（chunking前，including descriptions）
        self.chunks_document_text = self.key_value_storage_cls(
            namespace="chunks_document_text",
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于存储原始图像页
        self.chunks_document_image = self.key_value_storage_cls(
            namespace="chunks_document_image",
            global_config=asdict(self)
        )

        # 初始化存储类实例，用于存储文本块
        self.chunks_information_text = self.key_value_storage_cls(
            namespace="chunks_information_text",
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于存储图像块(image description)
        self.chunks_information_image = self.key_value_storage_cls(
            namespace="chunks_information_image",
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于融合信息（块）
        self.chunks_information_fusion = self.key_value_storage_cls(
            namespace="chunks_information_fusion",
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于融合信息（句子）
        self.chunks_information_sentences = self.key_value_storage_cls(
            namespace="chunks_information_sentences",
            global_config=asdict(self)
        )

        # 初始化存储类实例，用于存储文本块的嵌入向量
        self.embeddings_chunks = self.vector_storage_cls(
            namespace="embeddings_chunks" + '_' + self.text_encoder_name.rsplit('/', 1)[-1],
            embedding_dim=self.text_embedding_dim,
            embedding_model_name=global_config.TEXT_ENCODER_MODEL,
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于存储文本句的嵌入向量
        self.embeddings_sentences = self.vector_storage_cls(
            namespace="embeddings_sentences" + '_' + self.text_encoder_name.rsplit('/', 1)[-1],
            embedding_dim=self.text_embedding_dim,
            embedding_model_name=global_config.TEXT_ENCODER_MODEL,
            cosine_threshold=global_config.embeddings_sentences_cosine_threshold,
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于存储文本实体的嵌入向量
        self.embeddings_entities = self.vector_storage_cls(
            namespace="embeddings_entities" + '_' + self.text_encoder_name.rsplit('/', 1)[-1],
            embedding_dim=self.text_embedding_dim,
            embedding_model_name=global_config.TEXT_ENCODER_MODEL,
            cosine_threshold=global_config.embeddings_entities_cosine_threshold,
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于存储文本关系的嵌入向量
        self.embeddings_relationships = self.vector_storage_cls(
            namespace="embeddings_relationships" + '_' + self.text_encoder_name.rsplit('/', 1)[-1],
            embedding_dim=self.text_embedding_dim,
            embedding_model_name=global_config.TEXT_ENCODER_MODEL,
            global_config=asdict(self)
        )
        # 初始化存储类实例，用于存储多模态块的嵌入向量
        self.embeddings_save_path = os.path.join(self.working_dir,
                                                 'embeddings_multi_modal' + '_' + self.mm_encoder_name.rsplit('/', 1)[
                                                     -1] + '.pt')
        if os.path.exists(self.embeddings_save_path):
            # 使用 map_location 将存储的内容直接映射到指定设备
            self.embeddings_multi_modal = torch.load(
                self.embeddings_save_path,
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                weights_only=True
            ).float()
        else:
            self.embeddings_multi_modal = None

        # 初始化存储类实例，用于存储图
        self.knowledge_graph = self.graph_storage_cls(
            namespace="graph",
            global_config=asdict(self)
        )

        # 初始化llm存储实例，用于存放llm的响应
        self.llm_response_cache = self.key_value_storage_cls(
            namespace="llm_response_cache",
            global_config=asdict(self)
        )
        # 初始化mllm存储实例，用于存放mllm的响应
        self.mllm_response_cache = self.key_value_storage_cls(
            namespace="mllm_response_cache",
            global_config=asdict(self)
        )

    def insert_origin_page_image(self, page_images_information):
        document_image = dict()
        for page_image in page_images_information:
            document_image[compute_mdhash_id(page_image['image_path'], prefix="pi-")] = {
                'image_path': page_image['image_path'],
                'image_type': 'page_image',
                'page_idx': page_image['page_idx'] + 1,
            }

        # 筛选出需要添加的新文档ID
        _add_doc_keys = self.chunks_document_image.filter_keys(list(document_image.keys()))
        # 根据筛选结果更新新文档字典
        document_image = {k: v for k, v in document_image.items() if k in _add_doc_keys}
        if document_image:
            self.chunks_document_image.upsert(document_image)
            self.chunks_document_image.index_done_callback()

    def insert_text(self, string_or_strings, pages_idx=None):
        # 分块并插入文本块数据库
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        if pages_idx is None:
            pages_idx = [None] * len(string_or_strings)
        elif len(pages_idx) != len(string_or_strings):
            raise ValueError("pages_idx 长度必须与 string_or_strings 一致")

        # 插入新文档
        new_docs = {
            compute_mdhash_id(c.strip(), prefix="doc-"): {'content': c.strip(), 'page_idx': p}
            for c, p in zip(string_or_strings, pages_idx)
        }
        # 筛选出需要添加的新文档ID
        _add_doc_keys = self.chunks_document_text.filter_keys(list(new_docs.keys()))
        # 根据筛选结果更新新文档字典
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}

        if new_docs:
            # chunking every doc
            all_chunks = {}
            for doc_key, doc in new_docs.items():
                doc['content'] = doc['content'].strip()
                if len(doc['content']) == 0:
                    continue
                chunks = {
                    compute_mdhash_id(dp['content'], prefix="text-"): {
                        **dp,
                        'full_doc_id': doc_key,
                        'page_idx': doc['page_idx']
                    }
                    for dp in chunking_by_token_size(
                        doc['content'],
                        overlap_token_size=global_config.chunk_overlap_token_size,
                        max_token_size=global_config.chunk_max_token_size,
                        tiktoken_model=global_config.tiktoken_model_name,
                    )
                }
                all_chunks.update(chunks)

            # 筛选出需要添加的新片段ID
            _add_chunk_keys = self.chunks_information_text.filter_keys(list(all_chunks.keys()))
            # 根据筛选结果更新新片段字典
            inserting_chunks = {k: v for k, v in all_chunks.items() if k in _add_chunk_keys}
            # 如果没有新片段需要添加，记录日志并返回
            if not len(inserting_chunks):
                logger.warning(f"All chunks are already in the storage")
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")

            self.chunks_document_text.upsert(new_docs)
            self.chunks_document_text.index_done_callback()
            self.chunks_information_text.upsert(inserting_chunks)
            self.chunks_information_text.index_done_callback()

    def insert_image(self, content_list, images_path, page_information=None):
        for idx, c in enumerate(content_list):
            if len(c['img_path']) < 5:
                continue
            image_id = compute_mdhash_id(c['img_path'], 'doc-')
            if self.chunks_document_text.get_by_id(image_id) is not None:
                continue

            image_path = images_path + c['img_path']
            pdf_name = Path(images_path).parent.name
            pdf_page_path = self.working_dir + '/page_image/' + pdf_name + '-' + str(c['page_idx'] + 1) + '.png'
            complete_origin_image = {'input_type': 'image', 'input_path': pdf_page_path}
            complete_image = {'input_type': 'image', 'input_path': image_path}
            img_base = [complete_origin_image, complete_image]
            # if only little information, merge
            if page_information is not None:
                if page_information[c['page_idx'] + 1]['token_num'] < 100 and page_information[c['page_idx'] + 1][
                    'image_num'] < 2:
                    img_base = [complete_origin_image]
                    image_path = pdf_page_path

            content, caption, image_type, description, _, _ = image_summary_by_mllm(img_base,
                                                                                    hashing_kv=self.mllm_response_cache)

            # filter
            if len(description) < 5:
                continue

            complete_image = {image_id: {
                'type': c['type'],
                'image_id': image_id,
                'image_path': image_path,
                'page_idx': int(c['page_idx']) + 1,
                'caption': caption,
                'image_type': image_type,
                'description': description,
            }}

            caption_token_length = len(
                tiktoken.encoding_for_model(global_config.tiktoken_model_name).encode(caption))
            chunks = {
                compute_mdhash_id(dp['content'], prefix="img-"): {
                    'tokens': dp['tokens'] + caption_token_length,
                    'caption': caption,
                    'description': dp['content'],
                    'content': caption + '\n' + dp['content'],
                    'chunk_order_index': dp['chunk_order_index'],
                    'image_path': images_path + c['img_path'],
                    'image_type': image_type,
                    'full_doc_id': image_id,
                    'page_idx': int(c['page_idx']) + 1,
                    'table_body': c.get('table_body')
                }
                for dp in chunking_by_token_size(
                    description,
                    overlap_token_size=global_config.chunk_overlap_token_size,
                    max_token_size=global_config.chunk_max_token_size - min(caption_token_length, 100),
                    tiktoken_model=global_config.tiktoken_model_name,
                )
            }

            self.chunks_document_text.upsert(complete_image)
            self.chunks_information_image.upsert(chunks)
        self.chunks_document_text.index_done_callback()
        self.chunks_information_image.index_done_callback()

    def generate_embeddings(self, page_image_path_list):
        logger.info("Save image embeddings")
        self.embeddings_multi_modal = information_encode(self.mm_encoder, images=page_image_path_list, batch_size=32)
        torch.save(self.embeddings_multi_modal, self.embeddings_save_path)

    def retrieve_document(self, query, method="base", top_k=60, **kwargs):
        chunks_results, retrieve_results, retrieve_page_list, retrieve_page_scores = [], [], [], []
        doc_retriever = DocumentRetriever(encoder=self.mm_encoder[0], processor=self.mm_encoder[1],
                                          device=self.mm_encoder[0].device)

        if method == 'base':
            chunks_results = self.embeddings_chunks.query(query, top_k=top_k)
        elif method == 'colpali' or method == 'colqwen' or method == 'colqwen-multilingual':
            retrieve_page_list, retrieve_page_scores = doc_retriever.base_retrieve(query, self.embeddings_multi_modal,
                                                                                   top_k=top_k)
            last_part = self.working_dir.split('/')[-1]
            for (page, page_score) in zip(retrieve_page_list, retrieve_page_scores):
                chunks_results.append({
                    '__metrics__': float(page_score),
                    'page_idx': page,
                    'image_path': f"/page_image/{last_part}-{page}.png"
                })
            return chunks_results, retrieve_page_list, retrieve_page_scores

        for chunks_information in chunks_results:
            image_path = chunks_information.get('image_path', '')
            if method == 'base' and 'img-' in chunks_information['__id__']:
                chunk_image_information = self.chunks_information_image.get_by_id(chunks_information['__id__'])
                image_path = chunk_image_information['image_path']
            retrieve_results.append({
                'unit_ids': chunks_information['__id__'],
                'score': float(chunks_information['__metrics__']),
                'page_idx': chunks_information['page_idx'],
                'content': chunks_information['content'],
                'image_path': image_path
            })
            if chunks_information['page_idx'] not in retrieve_page_list:
                retrieve_page_list.append(chunks_information['page_idx'])
                retrieve_page_scores.append(chunks_information['__metrics__'])

        return retrieve_results, retrieve_page_list, retrieve_page_scores

    def answer(self, sample, retrieve_results, top_k=10, response_cache=None):
        page_idx = set()
        retrieve_information = []
        for retrieve_result in retrieve_results:
            if retrieve_result['page_idx'] not in page_idx:
                page_idx.add(retrieve_result['page_idx'])
            if len(page_idx) > top_k:
                break
            information = {'input_page_idx': retrieve_result['page_idx']}
            if 'image_path' in retrieve_result and len(retrieve_result['image_path']) > 0:
                information['input_type'] = 'image'
                if 'unit_ids' in retrieve_result:
                    chunk_image_information = self.chunks_information_image.get_by_id(retrieve_result['unit_ids'])
                    information['content'] = chunk_image_information['description']
                    information['input_path'] = self.working_dir + chunk_image_information['image_path']
                else:
                    information['input_path'] = self.working_dir + retrieve_result['image_path']
            else:
                information['input_type'] = 'text'
                chunk_text_information = self.chunks_information_text.get_by_id(retrieve_result['unit_ids'])
                information['content'] = chunk_text_information['content']
            retrieve_information.append(information)

        prompt = global_config.prompt_method
        retrieve_information = information_process_test(retrieve_information)
        # if 'plan' not in prompt:
        #     retrieve_information = information_process_test(retrieve_information)
        # else:
        #     retrieve_information = information_process(retrieve_information)
        return query_answer_by_mllm(sample['question'], retrieve_information,
                                    answer_prompt=prompt, hashing_kv=response_cache)
