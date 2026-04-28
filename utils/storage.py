import html
import networkx as nx
import igraph as ig
import numpy as np
import os
from dataclasses import dataclass
from typing import Union, Generic, TypeVar, cast, Any

from llm.encoder import init_model
from utils.file_utils import load_json, write_json
from utils.decorator_utils import lazy_property
from utils.vectordb import NanoVectorDB, ChromaVectorDB
from config.base_config import logger

# 用来定义泛型函数
T = TypeVar("T")


@dataclass
class StorageNameSpace:
    """
    存储命名空间类，用于管理存储操作。

    属性:
    namespace (str): 命名空间名称。
    global_config (dict): 全局配置信息。

    方法:
    index_done_callback: 在索引完成后提交存储操作。
    query_done_callback: 在查询完成后提交存储操作。
    """
    namespace: str
    global_config: dict

    def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """
    基础向量存储类，继承自 StorageNameSpace。

    属性:
    namespace (str): 命名空间名称。
    global_config (Dict[str, Any]): 全局配置信息。

    方法:
    query: 查询方法，具体函数在_storage.py中，下同。
    upsert: 插入或更新方法。
    """

    def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """
    基础键值存储类，继承自 StorageNameSpace。

    属性:
    namespace (str): 命名空间名称。
    global_config (Dict[str, any]): 全局配置信息。

    方法:
    all_keys: 获取所有键。
    get_by_id: 根据 ID 获取数据。
    get_by_ids: 根据多个 ID 获取数据。
    filter_keys: 筛选出不存在的键。
    upsert: 插入或更新数据。
    drop: 删除整个存储空间。
    """

    def all_keys(self) -> list[str]:
        raise NotImplementedError

    def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    def get_by_ids(
            self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError

    def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """
    基础图存储类，继承自 StorageNameSpace。

    属性:
    namespace (str): 命名空间名称。
    global_config (Dict[str, any]): 全局配置信息。

    方法:
    has_node: 判断节点是否存在。
    has_edge: 判断边是否存在。
    node_degree: 获取节点度数。
    edge_degree: 获取边的度数。
    get_node: 获取节点信息。
    get_edge: 获取边信息。
    get_node_edges: 获取节点的所有边。
    upsert_node: 插入或更新节点。
    upsert_edge: 插入或更新边。
    clustering: 进行图聚类。
    community_schema: 获取社区结构。
    embed_nodes: 对节点进行嵌入。
    """

    def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    def get_node_edges(
            self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    def clustering(self, algorithm: str):
        raise NotImplementedError

    def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Node embedding is not used in nano-graphrag.")


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self, ):
        # 初始化时，根据全局配置确定工作目录，以便获取文件完整路径
        working_dir = self.global_config["working_dir"]
        # 根据命名空间生成特定的 JSON 文件名，用于存储键值数据
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        # 加载存储的数据，如果文件不存在或为空，则初始化为空字典
        self._data = load_json(self._file_name) or {}
        # 打印日志，显示加载的数据条数
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    # 获取所有的键列表
    def all_keys(self) -> list[str]:
        return list(self._data.keys())

    # 索引操作完成后，将当前数据写入 JSON 文件
    def index_done_callback(self):
        write_json(self._data, self._file_name)

    # 通过 ID 获取数据
    def get_by_id(self, id):
        return self._data.get(id, None)

    def get_by_ids(self, ids, fields=None):
        """
        根据ID列表获取数据项。

        参数:
        ids (list): 需要获取数据的ID列表。
        fields (list, 可选): 限制返回数据中的字段。如果未提供，默认为None，将返回完整数据项。

        返回:
        list: 包含按指定ID列表顺序排列的数据项的列表。如果某些ID未找到数据项，则相应位置为None。
        """
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                # 如果数据项存在，并且ID在_data字典中，则构建一个仅包含fields中字段的新字典
                {k: v for k, v in self._data[id].items() if k in fields}
                # 检查_id是否在数据集中，以避免KeyError
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    def get_data(self) -> dict:
        return self._data

    # 过滤出不在数据存储中的键列表
    def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    # 插入或更新数据
    def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    # 清空当前存储数据
    def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    embedding_dim: int = 128
    # 余弦相似度阈值，决定返回的结果质量
    cosine_threshold: float = 0.2
    embedding_model_name: str = None

    @lazy_property
    def embedding_model(self):
        return init_model(model_name=self.embedding_model_name)

    def __post_init__(self):
        # 初始化向量数据库存储文件和嵌入配置
        self._client_file_name = os.path.join(self.global_config["working_dir"], f"vdb_{self.namespace}.json")
        self._max_batch_size = self.global_config["embedding_batch_num"]
        # 初始化向量数据库客户端（NanoVectorDB），并设置嵌入维度
        self._client = NanoVectorDB(self.embedding_dim, storage_file=self._client_file_name)
        # 从全局配置中获取查询的相似度阈值，或使用默认值
        self.cosine_threshold = self.global_config.get("query_threshold", self.cosine_threshold)

    def upsert(self, data: dict[str, dict]):
        """
        插入或更新向量数据。

        该方法用于将字典形式的数据插入或更新到向量数据库中。数据首先被转换成适合插入的格式，
        然后分批处理，以避免一次性插入过多数据导致的性能问题。之后，使用异步方式计算各批次数据的嵌入向量，
        并将这些向量附加到数据条目中，最后调用客户端的插入或更新方法完成操作。

        参数:
        data: dict[str, dict] - 一个字典，键是数据的唯一标识，值是包含实际数据内容的字典。

        返回:
        插入或更新操作的结果。
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        # 将数据转换为适合插入的列表，并提取内容
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        information = [v["content"] for v in data.values()]
        # 计算数据的嵌入向量
        embeddings = self.embedding_model.encode(information)
        # 将计算得到的嵌入向量附加到每个数据条目中
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        # 调用客户端的插入或更新方法，完成数据的插入或更新
        results = self._client.upsert(datas=list_data)
        return results

    def query(self, query: str, top_k=5, min_max_norm=False):
        """
        根据提供的查询字符串获取最相关的文档。

        此异步方法使用预训练的embedding函数将查询转换为嵌入表示，
        然后在嵌入索引中搜索与查询最相似的文档。

        参数:
        - query: str，用户查询的字符串。
        - top_k: int，返回最相关的文档数量，默认为5。

        返回:
        - 一个列表，包含最相关的文档及其与查询的相似度距离。
        """
        if isinstance(query, str):
            embedding = self.embedding_model.encode([query])[0]
        else:
            embedding = query  # 假设已经是嵌入向量
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_threshold,
            min_max_norm=min_max_norm
        )
        # 整理结果，添加文档id和距离信息
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    def queries(self, queries: list, top_k=5, min_max_norm=False):
        """
        同上，但是可以兼容多个query
        """
        if len(queries) == 0:
            return []
        embedding = self.embedding_model.encode(queries)
        results = self._client.queries(
            queries=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_threshold,
            min_max_norm=min_max_norm
        )
        # 整理结果，添加文档id和距离信息
        final_results = []
        for result_list in results:
            result_l = [
                {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in result_list
            ]
            final_results.append(result_l)
        return final_results

    def get_all(self):
        return self._client.get_all()

    def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    # 加载并返回一个NetworkX图，存储格式为graphml。
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            return nx.read_graphml(file_name)
        return None

    # 将NetworkX图写入graphml文件。
    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        """返回图的最大连通分量，并以稳定的方式排序节点和边。

        参数:
            graph (nx.Graph): 输入的 NetworkX 图。

        返回:
            nx.Graph: 输入图的最大连通分量，以稳定方式排序。
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        """
        确保无向图以相同的关系读取时始终相同。

        参数:
        graph (nx.Graph): 输入的网络图。

        返回:
        nx.Graph: 经过稳定处理的网络图。
        """
        # 根据输入图的类型初始化一个新的图实例
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        # 对节点进行排序，以确保节点的添加顺序一致
        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])
        # 向新图中添加排序后的节点
        fixed_graph.add_nodes_from(sorted_nodes)
        # 将边数据存储到列表中以便后续处理
        edges = list(graph.edges(data=True))
        # 如果图不是有向图，则对边进行排序，以确保边的顺序一致
        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        # 定义获取边的键的函数，用于后续边的排序
        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        # 对边进行排序
        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))
        # 向新图中添加排序后的边
        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        """
        初始化函数，用于加载图数据并初始化相关属性。

        该函数首先根据全局配置中的工作目录和实例的命名空间来确定graphml文件的路径。
        然后尝试从该路径加载已存在的图数据。如果图数据存在，则使用NetworkXStorage加载，
        并记录日志信息包括图的节点数和边数。如果图数据不存在，则初始化一个新的无向图。
        最后，初始化两个算法字典，分别用于图的聚类算法和节点嵌入算法。
        """
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    # 将当前存储的图写入到GraphML文件中
    def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    def has_node(self, node_id: str) -> bool:
        """
        异步检查图中是否存在指定节点。下面几个函数类似，就不做标注了，都是调用NetWorkX的函数。

        该方法主要用于确定图结构中是否包含特定的节点。它通过调用底层图对象的has_node方法，
        以高效的方式查询节点是否存在。

        参数:
        node_id (str): 要检查的节点的唯一标识符。

        返回:
        bool: 如果图中存在该节点，则返回True，否则返回False。
        """
        return self._graph.has_node(node_id)

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    # 获取指定节点的度数。
    def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    # 计算两个节点的度数之和
    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    # 根据指定的算法进行节点嵌入。
    def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        from graspologic import embed

        embeddings, nodes = embed.node2vec_embed(
            self._graph,
            **self.global_config["node2vec_params"],
        )

        nodes_ids = [self._graph.nodes[node_id]["id"] for node_id in nodes]
        return embeddings, nodes_ids


@dataclass
class IGraphStorage(BaseGraphStorage):
    def __post_init__(self):
        self._graphml_file = os.path.join(
            self.global_config["working_dir"],
            f"graph_{self.namespace}.graphml"
        )

        if os.path.exists(self._graphml_file) and os.path.getsize(self._graphml_file) > 0:
            logger.info(f"Loading igraph graph from {self._graphml_file}")
            self._graph = ig.Graph.Read_GraphML(self._graphml_file)
            logger.info(f"Loaded graph with {len(self._graph.vs)} nodes and {len(self._graph.es)} edges")
        else:
            self._graph = ig.Graph(directed=False)

        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    def index_done_callback(self):
        logger.info(f"Writing igraph graph with {len(self._graph.vs)} nodes and {len(self._graph.es)} edges")
        self._graph.write_graphml(self._graphml_file)

    def has_node(self, node_id: str) -> bool:
        try:
            return node_id in self._graph.vs["name"]
        except KeyError:
            return False

    def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        try:
            source_idx = self._graph.vs.find(name=source_node_id).index
            target_idx = self._graph.vs.find(name=target_node_id).index
            return self._graph.get_eid(source_idx, target_idx, directed=False, error=False) != -1
        except (ValueError, KeyError):
            return False

    def node_degree(self, node_id: str) -> int:
        try:
            vertex = self._graph.vs.find(name=node_id)
            return self._graph.degree(vertex)
        except (ValueError, KeyError):
            return 0

    def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self.node_degree(src_id) + self.node_degree(tgt_id)

    def get_node(self, node_id: str) -> Union[dict, None]:
        try:
            vertex = self._graph.vs.find(name=node_id)
            return dict(vertex.attributes())
        except (ValueError, KeyError):
            return None

    def get_edge(self, source_node_id: str, target_node_id: str) -> Union[dict, None]:
        try:
            source_idx = self._graph.vs.find(name=source_node_id).index
            target_idx = self._graph.vs.find(name=target_node_id).index
            eid = self._graph.get_eid(source_idx, target_idx, directed=False, error=False)
            if eid != -1:
                return dict(self._graph.es[eid].attributes())
            return None
        except (ValueError, KeyError):
            return None

    def get_node_edges(self, source_node_id: str) -> Union[list[tuple[str, str]], None]:
        if not self.has_node(source_node_id):
            return None

        try:
            source_idx = self._graph.vs.find(name=source_node_id).index
            edges = []
            for edge in self._graph.incident(source_idx):
                edge_obj = self._graph.es[edge]
                source_vertex = self._graph.vs[self._graph.es[edge].source]
                target_vertex = self._graph.vs[self._graph.es[edge].target]
                edge_info = {
                    "source": source_vertex["name"],
                    "target": target_vertex["name"],
                    "edge_id": edge,
                    "attributes": dict(edge_obj.attributes())
                }
                edges.append(edge_info)
            return edges
        except (ValueError, KeyError):
            return None

    def upsert_node(self, node_id: str, node_data: dict[str, str]):
        if not self.has_node(node_id):
            # 添加新节点
            self._graph.add_vertex(name=node_id, **node_data)
        else:
            # 更新现有节点
            vertex = self._graph.vs.find(name=node_id)
            for key, value in node_data.items():
                vertex[key] = value

    def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]):
        # 确保节点存在
        if not self.has_node(source_node_id):
            self._graph.add_vertex(name=source_node_id)
        if not self.has_node(target_node_id):
            self._graph.add_vertex(name=target_node_id)

        source_idx = self._graph.vs.find(name=source_node_id).index
        target_idx = self._graph.vs.find(name=target_node_id).index

        # 查找边
        eid = self._graph.get_eid(source_idx, target_idx, directed=False, error=False)

        if eid == -1:
            # 添加新边
            self._graph.add_edge(source_idx, target_idx, **edge_data)
        else:
            # 更新现有边
            for key, value in edge_data.items():
                self._graph.es[eid][key] = value

    def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return self._node_embed_algorithms[algorithm]()

    def _node2vec_embed(self):
        from graspologic import embed

        # 将igraph图转换为networkx图进行嵌入
        nx_graph = self._graph.to_networkx()
        embeddings, nodes = embed.node2vec_embed(
            nx_graph,
            **self.global_config.get("node2vec_params", {}),
        )

        # 获取节点名称（按照networkx图的节点顺序）
        node_names = [str(node) for node in nodes]  # networkx节点可能是整数索引

        return embeddings, node_names

    def clear_graph(self) -> None:
        logger.info(f"Clearing igraph graph with {len(self._graph.vs)} nodes and {len(self._graph.es)} edges")
        self._graph = ig.Graph(directed=False)


@dataclass
class ChromaVectorDBStorage(BaseVectorStorage):
    """
    基于 Chroma 的向量存储类，结构和 NanoVectorDBStorage 一致。
    可与 NanoVectorDBStorage 互换使用。
    """

    cosine_threshold: float = 0.2
    embedding_model_name: str = None
    in_memory: bool = False

    def __post_init__(self):
        # 初始化存储路径和模型配置
        self._persist_dir = os.path.join(self.global_config["working_dir"], f"chroma_{self.namespace}")
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self.cosine_threshold = self.global_config.get("query_threshold", self.cosine_threshold)
        self.embedding_model = init_model(model_name=self.embedding_model_name)

        # 初始化 Chroma 客户端
        self._client = ChromaVectorDB(
            collection_name=self.namespace,
            persist_directory=self._persist_dir,
            metric="cosine",
            in_memory=self.in_memory,
        )
        logger.info(f"ChromaVectorDBStorage initialized for {self.namespace}")

    def upsert(self, data: dict[str, dict]):
        """
        插入或更新数据。使用 embedding_model 计算嵌入后存入 Chroma。
        """
        logger.info(f"Upserting {len(data)} vectors into Chroma collection: {self.namespace}")
        if not len(data):
            logger.warning("Empty data input for upsert.")
            return []

        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        embeddings = self.embedding_model.encode(contents)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    def query(self, query: str, top_k: int = 5, min_max_norm: bool = False):
        """
        根据查询字符串检索最相似文档。
        """
        if isinstance(query, str):
            embedding = self.embedding_model.encode([query])[0]
        else:
            embedding = query  # 已是向量
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    def queries(self, queries: list, top_k: int = 5, min_max_norm: bool = False):
        """
        支持多个查询同时执行。
        """
        if len(queries) == 0:
            return []
        embeddings = self.embedding_model.encode(queries)
        all_results = []
        for emb in embeddings:
            results = self._client.query(
                query=emb,
                top_k=top_k,
                better_than_threshold=self.cosine_threshold,
            )
            all_results.append([
                {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]}
                for dp in results
            ])
        return all_results

    def get_all(self):
        return self._client.get_all()

    def delete(self, ids: list[str]):
        self._client.delete(ids=ids)

    def index_done_callback(self):
        self._client.persist()
