import numpy as np
import torch
from colpali_engine import ColPali, ColPaliProcessor


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val


class DocumentRetriever:

    def __init__(self, encoder: ColPali, processor: ColPaliProcessor, device: torch.device, batch_size=512):
        self.encoder = encoder
        self.processor = processor
        self.device = device
        self.batch_size = batch_size

    def compute_scores(self, query, all_embeds):
        queries = self.processor.process_queries(queries=[query]).to(self.device)
        query_embeds = self.encoder(**queries)

        all_scores = []
        for idx in range(0, all_embeds.shape[0], self.batch_size):
            batch_embeds = all_embeds[idx: idx + self.batch_size]
            batch_embeds = batch_embeds.to(device=self.device, dtype=query_embeds.dtype)

            with torch.no_grad():
                tmp_scores = self.processor.score_multi_vector(query_embeds.real, batch_embeds.real)
                if len(tmp_scores.shape) > 1:
                    tmp_scores = tmp_scores[0]

            all_scores.append(tmp_scores)

        scores = torch.cat(all_scores, dim=0).cpu()
        del all_scores, queries, query_embeds
        return scores

    def compute_batch_scores(self, qs, ps):
        """
        计算批量多向量查询和文档之间的分数
        qs: [batch_size_q, seq_len_q, embed_dim]
        ps: [batch_size_p, seq_len_p, embed_dim]
        返回: [batch_size_q, batch_size_p]
        """
        similarity_matrix = torch.einsum("bnd,csd->bcns", qs, ps)
        max_similarities = similarity_matrix.max(dim=3)[0]
        scores = max_similarities.sum(dim=2)
        return scores.to(torch.float32)

    def base_retrieve(self, query, all_embeds, top_k=60):
        scores = self.compute_scores(query, all_embeds)
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k].tolist()
        top_scores = scores[top_indices].tolist()
        # 1-indexed
        return [idx + 1 for idx in top_indices], top_scores

    def base_retrieve_with_data(self, query, all_embeds, page_image_information, top_k=60, min_max_norm=False):
        scores = self.compute_scores(query, all_embeds)
        top_indices = scores.argsort(dim=-1, descending=True)[:top_k].tolist()
        top_scores = scores[top_indices].tolist()
        if min_max_norm:
            top_scores = min_max_normalize(top_scores)
        page_image_ids = list(page_image_information)
        page_image_data = []
        for top_ind, top_score in zip(top_indices, top_scores):
            page_image_data.append({
                '__id__': page_image_ids[top_ind],
                '__metrics__': top_score,
                'page_idx': top_ind + 1
            })
        return page_image_data
