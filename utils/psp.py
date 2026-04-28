# psp_query_system.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict


# ---------------------------
# PSPIndex 实现（简化无 AET）
# ---------------------------
class PSPIndex:
    def __init__(self, data: np.ndarray,
                 K: int = 50, R: int = 40, S: int = 5, alpha_deg: float = 60.0,
                 sn_clusters: int = 32, sn_m: int = 512):
        self.data = data.astype(np.float32)
        self.n, self.d = self.data.shape
        self.K = min(K, self.n - 1)
        self.R = R
        self.S = S
        self.alpha_deg = alpha_deg
        self.alpha_cos_threshold = np.cos(np.radians(alpha_deg))
        self.sn_clusters = min(sn_clusters, max(1, self.n // 10))
        self.sn_m = min(sn_m, self.n)
        self.graph: List[List[int]] = [[] for _ in range(self.n)]
        self.nav_ivf: Dict[int, List[int]] = {}
        self.cluster_centers = None
        self.rng = np.random.RandomState(42)

    def build(self):
        print("Building PSP index...")
        self._build_base_knng()
        self._edge_refinement()
        self._spherical_navigation()
        print("PSP index built successfully.\n")

    def _build_base_knng(self):
        neigh = NearestNeighbors(n_neighbors=self.K + 1, algorithm="auto", metric="euclidean")
        neigh.fit(self.data)
        _, indices = neigh.kneighbors(self.data)
        for i in range(self.n):
            neighs = [int(x) for x in indices[i, 1:self.K + 1]]
            self.graph[i] = neighs[:self.R]

    def _edge_refinement(self):
        G_new = [set(neis) for neis in self.graph]
        for n in range(self.n):
            first_neighbors = self.graph[n]
            c_set = set()
            for p in first_neighbors:
                for q in self.graph[p]:
                    if q != n:
                        c_set.add(q)
            c_set.discard(n)
            c_list = list(c_set)
            if not c_list:
                continue
            n_vec = self.data[n]
            ips = np.dot(self.data[c_list], n_vec)
            order = np.argsort(-ips)
            c_sorted = [c_list[i] for i in order]
            E_n = []
            E_n.append(c_sorted[0])
            for cand in c_sorted[1:]:
                if len(E_n) >= self.S:
                    break
                cos_val = np.dot(n_vec, self.data[cand]) / (
                            np.linalg.norm(n_vec) * np.linalg.norm(self.data[cand]) + 1e-9)
                if cos_val > self.alpha_cos_threshold:
                    continue
                E_n.append(cand)
            for v in E_n:
                if v != n:
                    G_new[n].add(v)
            all_neis = list(G_new[n])
            if len(all_neis) > self.R:
                ips2 = np.dot(self.data[all_neis], n_vec)
                keep_idx = np.argsort(-ips2)[:self.R]
                G_new[n] = set([all_neis[i] for i in keep_idx])
        self.graph = [list(s) for s in G_new]

    def _spherical_navigation(self):
        norms = np.linalg.norm(self.data, axis=1)
        normed = self.data / (norms[:, None] + 1e-9)
        kmeans = KMeans(n_clusters=self.sn_clusters, random_state=0, n_init=8)
        labels = kmeans.fit_predict(normed)
        centers = kmeans.cluster_centers_
        self.cluster_centers = centers
        invlists = {i: [] for i in range(self.sn_clusters)}
        m_per = max(1, self.sn_m // self.sn_clusters)
        for cl in range(self.sn_clusters):
            idxs = np.where(labels == cl)[0]
            if idxs.size == 0:
                continue
            probs = norms[idxs]
            if probs.sum() == 0:
                probs = np.ones_like(probs)
            probs = probs / probs.sum()
            sel_count = min(m_per, idxs.size)
            chosen = self.rng.choice(idxs, size=sel_count, replace=False, p=probs)
            invlists[cl] = list(map(int, chosen))
        self.nav_ivf = invlists

    def _find_closest_nav_cluster(self, q: np.ndarray) -> int:
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        sims = np.dot(self.cluster_centers, q_norm)
        return int(np.argmax(sims))

    def search(self, q: np.ndarray, k: int = 10, ls: int = 200) -> List[Tuple[int, float]]:
        q = q.astype(np.float32)
        cl_idx = self._find_closest_nav_cluster(q)
        pool = self.nav_ivf.get(cl_idx, [])
        if len(pool) == 0:
            pool = list(self.rng.choice(self.n, size=min(10, self.n), replace=False))
        visited = set()
        Q = []
        R = []
        for p in pool:
            score = np.dot(self.data[p], q)
            Q.append((score, p))
        Q.sort(reverse=True)
        while Q:
            score, p = Q.pop(0)
            if p in visited:
                continue
            visited.add(p)
            ip_val = np.dot(self.data[p], q)
            if len(R) < k:
                R.append((ip_val, p))
                R.sort(reverse=True)
            else:
                if ip_val > R[-1][0]:
                    R[-1] = (ip_val, p)
                    R.sort(reverse=True)
            for nbr in self.graph[p]:
                if nbr not in visited:
                    score_nbr = np.dot(self.data[nbr], q)
                    Q.append((score_nbr, nbr))
            Q.sort(reverse=True)
            if len(Q) > ls:
                Q = Q[:ls]
        return [(p, float(s)) for s, p in R]
