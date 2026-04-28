import os
import json
import numpy as np
import base64
import hashlib
import chromadb
from dataclasses import dataclass, asdict
from typing import Union, TypeVar, Literal, Callable, TypedDict, List, Dict, Any, Optional
from locale import normalize
from numpy import copy
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config.base_config import logger
from utils.psp import PSPIndex

# 用来定义泛型函数
T = TypeVar("T")
f_ID = "__id__"
f_VECTOR = "__vector__"
f_METRICS = "__metrics__"
Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)
Float = np.float32
ConditionLambda = Callable[[Data], bool]


def array_to_buffer_string(array: np.ndarray) -> str:
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name) -> Union[DataBase, None]:
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val


def min_max_normalize_rows(x):
    """对每一行独立进行归一化"""
    min_vals = np.min(x, axis=1, keepdims=True)  # 每行的最小值
    max_vals = np.max(x, axis=1, keepdims=True)  # 每行的最大值
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (x - min_vals) / range_vals


@dataclass
class NanoVectorDB:
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    storage_file: str = "nano-vectordb.json"
    _psp_index = None

    def pre_process(self):
        self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        storage: DataBase = load_storage(self.storage_file) or default_storage
        assert (
                storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        self.usable_metrics = {
            "cosine": self._cosine_query,
            "psp": self._psp_query,
        }
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def get_additional_data(self):
        return self.__storage.get("additional_data", {})

    def store_additional_data(self, **kwargs):
        self.__storage["additional_data"] = kwargs

    def upsert(self, datas: list[Data]):
        _index_datas = {
            data.get(f_ID, hash_ndarray(data[f_VECTOR])): data for data in datas
        }
        if self.metric == "cosine":
            for v in _index_datas.values():
                v[f_VECTOR] = normalize(v[f_VECTOR])
        report_return = {"update": [], "insert": []}
        for i, already_data in enumerate(self.__storage["data"]):
            if already_data[f_ID] in _index_datas:
                update_d = _index_datas.pop(already_data[f_ID])
                self.__storage["matrix"][i] = update_d[f_VECTOR].astype(Float)
                del update_d[f_VECTOR]
                self.__storage["data"][i] = update_d
                report_return["update"].append(already_data[f_ID])
        if len(_index_datas) == 0:
            return report_return
        report_return["insert"].extend(list(_index_datas.keys()))
        new_matrix = np.array(
            [data[f_VECTOR] for data in _index_datas.values()], dtype=Float
        )
        new_datas = []
        for new_k, new_d in _index_datas.items():
            del new_d[f_VECTOR]
            new_d[f_ID] = new_k
            new_datas.append(new_d)
        self.__storage["data"].extend(new_datas)
        self.__storage["matrix"] = np.vstack([self.__storage["matrix"], new_matrix])
        return report_return

    def get(self, ids: list[str]):
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def get_all_by_ids(self, ids: list[str]):
        result_data = []
        result_matrix = []
        for id in ids:
            found = False
            for i, data in enumerate(self.__storage["data"]):
                if data[f_ID] == id:
                    result_data.append(data)
                    result_matrix.append(self.__storage["matrix"][i])
                    found = True
                    break
            if not found:
                result_data.append({'__id__': id})
                result_matrix.append(np.zeros_like(self.__storage["matrix"][0]))
        return result_data, np.array(result_matrix)

    def get_all(self):
        return self.__storage["data"], self.__storage["matrix"]

    def delete(self, ids: list[str]):
        ids = set(ids)
        left_data = []
        delete_index = []
        for i, data in enumerate(self.__storage["data"]):
            if data[f_ID] in ids:
                delete_index.append(i)
                ids.remove(data[f_ID])
            else:
                left_data.append(data)
        self.__storage["data"] = left_data
        self.__storage["matrix"] = np.delete(self.__storage["matrix"], delete_index, axis=0)

    def save(self):
        storage = {
            **self.__storage,
            "matrix": array_to_buffer_string(self.__storage["matrix"]),
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(storage, f, ensure_ascii=False)

    def __len__(self):
        return len(self.__storage["data"])

    def query(
            self,
            query: np.ndarray,
            top_k: int = 10,
            better_than_threshold: float = None,
            filter_lambda: ConditionLambda = None,
            min_max_norm=False
    ) -> list[dict]:
        return self._cosine_query(query, top_k, better_than_threshold, filter_lambda=filter_lambda, min_max_norm=min_max_norm)

    def queries(
            self,
            queries: np.ndarray,
            top_k: int = 10,
            better_than_threshold: float = None,
            filter_lambda: ConditionLambda = None,
            min_max_norm=False
    ) -> list[dict]:
        return self._cosine_queries(queries, top_k, better_than_threshold, filter_lambda=filter_lambda, min_max_norm=min_max_norm)

    def _cosine_query(
            self,
            query: np.ndarray,
            top_k: int,
            better_than_threshold: float,
            filter_lambda: ConditionLambda = None,
            min_max_norm=False
    ):
        query = normalize(query)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]
        scores = np.dot(use_matrix, query)

        threshold_scores = copy(scores)
        if min_max_norm:
            scores = min_max_normalize(scores)

        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]
        sort_abs_index = filter_index[sort_index]
        results = []
        for abs_i, rel_i in zip(sort_abs_index, sort_index):
            if (
                    better_than_threshold is not None
                    and threshold_scores[rel_i] < better_than_threshold
            ):
                break
            results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i], 'embedding': use_matrix[rel_i]})
        return results

    def _cosine_queries(
            self,
            queries: np.ndarray,  # 改为复数形式，表示可以处理多个查询
            top_k: int,
            better_than_threshold: float,
            filter_lambda: ConditionLambda = None,
            min_max_norm: bool = False,
    ):
        # 归一化所有查询向量
        queries = normalize(queries)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]

        scores = np.dot(queries, use_matrix.T)
        threshold_scores = copy(scores)
        if min_max_norm:
            scores = min_max_normalize_rows(scores)

        all_results = []

        # 对每个查询向量分别处理
        for i, query_scores in enumerate(scores):
            # 获取当前查询的top_k结果
            sort_index = np.argsort(query_scores)[-top_k:]
            sort_index = sort_index[::-1]  # 从大到小排序
            sort_abs_index = filter_index[sort_index]

            results = []
            for abs_i, rel_i in zip(sort_abs_index, sort_index):
                if better_than_threshold is not None and threshold_scores[i][rel_i] < better_than_threshold:
                    break
                results.append({
                    **self.__storage["data"][abs_i],
                    f_METRICS: query_scores[rel_i],
                    'embedding': use_matrix[rel_i],
                    'query_index': i  # 添加查询索引以便区分不同查询的结果
                })
            all_results.append(results)

        return all_results

    # ---- PSP 构建 ----
    def build_psp(self):
        self._psp_index = PSPIndex(self.__storage["matrix"])
        self._psp_index.build()

    # ---- PSP 查询 ----
    def _psp_query(self, query: np.ndarray, top_k: int, better_than_threshold: float, filter_lambda=None):
        if self._psp_index is None:
            raise RuntimeError("PSP index not built. Call build_psp() first.")
        query = query.astype(np.float32)
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            filter_index = np.array([i for i, d in enumerate(self.__storage["data"]) if filter_lambda(d)])
            use_matrix = self.__storage["matrix"][filter_index]
        # Run PSP search
        results_raw = self._psp_index.search(query, k=top_k)
        results = []
        for idx, score in results_raw:
            if better_than_threshold is not None and score < better_than_threshold:
                break
            results.append({
                **self.__storage["data"][idx],
                f_METRICS: score,
                "embedding": use_matrix[idx]
            })
        return results


class ChromaVectorDB:
    """
    简洁高兼容性的 Chroma 向量数据库封装。
    可与 NanoVectorDB 类互换使用。
    """

    def __init__(
            self,
            collection_name: str = "default_collection",
            persist_directory: str = "./chroma_storage",
            metric: Literal["cosine", "l2", "ip"] = "cosine",
            embedding_function: Callable = None,
            in_memory: bool = False,
    ):
        self.metric = metric
        self.persist_directory = None if in_memory else persist_directory

        self.client = chromadb.Client(
            Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
        )

        # 默认embedding函数（可替换为OpenAI, HuggingFace等）
        if embedding_function is None:
            embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.metric},
            embedding_function=embedding_function,
        )
        logger.info(f"ChromaVectorDB initialized: {collection_name}, metric={metric}")

    def upsert(self, datas: List[dict]):
        ids, embeddings, metadatas = [], [], []
        for d in datas:
            ids.append(d.get(f_ID) or str(hash(np.array(d[f_VECTOR]).tobytes())))
            embeddings.append(np.array(d[f_VECTOR], dtype=Float).tolist())
            md = {k: v for k, v in d.items() if k not in [f_ID, f_VECTOR]}
            metadatas.append(md)

        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Upserted {len(ids)} items into ChromaDB.")
        return {"insert": ids}

    def query(
            self,
            query: Union[np.ndarray, list],
            top_k: int = 10,
            better_than_threshold: float = None,
            filter_lambda: ConditionLambda = None,
    ) -> List[dict]:
        query = np.array(query, dtype=Float).reshape(1, -1)
        results = self.collection.query(query_embeddings=query.tolist(), n_results=top_k)

        hits = []
        for i in range(len(results["ids"][0])):
            _id = results["ids"][0][i]
            _score = results["distances"][0][i]
            _meta = results["metadatas"][0][i]

            # Threshold过滤
            if better_than_threshold is not None and _score < better_than_threshold:
                continue

            # Lambda过滤
            if filter_lambda is not None and not filter_lambda(_meta):
                continue

            hits.append({f_ID: _id, f_METRICS: _score, **_meta})
        return hits

    def delete(self, ids: List[str]):
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} items from ChromaDB.")

    def get(self, ids: List[str]):
        res = self.collection.get(ids=ids)
        data = []
        for i, _id in enumerate(res["ids"]):
            if not _id:
                continue
            d = res["metadatas"][i]
            d[f_ID] = _id
            d[f_VECTOR] = np.array(res["embeddings"][i], dtype=Float)
            data.append(d)
        return data

    def get_all(self):
        res = self.collection.get()
        data, matrix = [], []
        for i, _id in enumerate(res["ids"]):
            d = res["metadatas"][i]
            d[f_ID] = _id
            data.append(d)
            matrix.append(np.array(res["embeddings"][i], dtype=Float))
        return data, np.array(matrix)

    def persist(self):
        if self.persist_directory:
            self.client.persist()
            logger.info(f"ChromaDB persisted to {self.persist_directory}.")

    def __len__(self):
        return self.collection.count()
