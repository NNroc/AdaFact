import re
import tiktoken
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter

from utils.base_utils import remove_null_chars


def merge_strings(str1, str2):
    len1 = len(str1)
    len2 = len(str2)
    max_overlap = min(len1, len2)
    for k in range(max_overlap, 0, -1):
        if str1[-k:] == str2[:k]:
            return str1 + str2[k:]
    return str1 + str2


def chunking_by_token_size(content: str, overlap_token_size=50, max_token_size=600, tiktoken_model="gpt-4o"):
    encoder = tiktoken.encoding_for_model(tiktoken_model)
    results, chunk_index = [], 0

    md_splitter = MarkdownTextSplitter.from_tiktoken_encoder(
        model_name=tiktoken_model,
        chunk_size=max_token_size,
        chunk_overlap=0,
    )
    markdown_chunks = md_splitter.create_documents([content])

    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=tiktoken_model,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )

    for doc in markdown_chunks:
        text = doc.page_content.strip()
        if not text:
            continue

        token_len = len(encoder.encode(text))
        if token_len > max_token_size:
            # secondary segmentation of long paragraphs
            sub_docs = token_splitter.create_documents([text])
            for sub_doc in sub_docs:
                results.append({
                    "tokens": len(encoder.encode(sub_doc.page_content)),
                    "content": sub_doc.page_content.strip(),
                    "chunk_order_index": chunk_index,
                    "title_path": doc.metadata.get("title_path", []),
                })
                chunk_index += 1
        else:
            # keep short paragraphs
            results.append({
                "tokens": token_len,
                "content": text,
                "chunk_order_index": chunk_index,
                "title_path": doc.metadata.get("title_path", []),
            })
            chunk_index += 1

    return results


def filter_spacy_content_when_fusion(text):
    """
    过滤可能被spacy误识别的内容
    """
    text = re.sub(r'```[\s\S]*?```', '', text)  # 代码块
    text = re.sub(r'<!--[\s\S]*?-->', '', text)  # HTML注释
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)  # 多行注释
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 图片链接

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    text = re.sub(r'`[^`]*`', '  ', text)  # 行内代码
    text = re.sub(r'<[^>]+>', '  ', text)  # HTML标签
    text = re.sub(r'^\|.*\|$', '  ', text)  # 表格标记

    text = re.sub(r'\$\$[\s\S]*?\$\$', '', text)  # 多行公式
    math_exprs = re.findall(r'\$[^$]+\$', text)
    for expr in math_exprs:
        # 如果表达式包含运算符，则替换为空
        if len(expr) > 30:
            text = text.replace(expr, '  ')

    # 去掉标题
    text = re.sub(r'#+ ', '', text)
    # 去掉公式
    text = re.sub(r'(?<!\\)\$(.+?)(?<!\\)\$', '', text)  # 行内公式
    # 去掉其它Markdown标记
    text = re.sub(r'\*|\_', '', text)

    text = re.sub(r'https?://\S+', '  ', text)
    text = re.sub(r'\b\d{10,}\b', '  ', text)  # 长数字序列

    # 过滤奇怪的空字符
    text = remove_null_chars(text)
    text = text.strip()
    return text


def split_content(content_list):
    content_image_list = []
    n = len(content_list)

    def collect_surrounding_texts(idx, k=2):
        """收集 idx 位置的最多 k 个上文文本和 k 个下文文本，返回按时间顺序拼接的字符串。"""
        prev_texts = []
        j = idx - 1
        while j >= 0 and len(prev_texts) < k:
            item = content_list[j]
            if item['page_idx'] != content_list[idx]['page_idx']:
                j -= 1
                continue
            if item.get('type') == 'text' and len(item.get('text', '')) > 5:
                prev_texts.append(item['text'])
            j -= 1
        prev_texts.reverse()

        next_texts = []
        j = idx + 1
        while j < n and len(next_texts) < k:
            item = content_list[j]
            if item['page_idx'] != content_list[idx]['page_idx']:
                j += 1
                continue
            if item.get('type') == 'text' and len(item.get('text', '')) > 5:
                next_texts.append(item['text'])
            j += 1

        parts = []
        if prev_texts:
            parts.extend(prev_texts)
        if next_texts:
            parts.extend(next_texts)
        return "\n".join(parts)

    for i, c in enumerate(content_list):
        # 保持原有行为：图片/表格根据 img_path 长度判断是否加入 content_image_list
        if (c.get('type') == 'image' or c.get('type') == 'table') and len(c.get('img_path', '')) > 7:
            surrounding = collect_surrounding_texts(i, k=1)
            c['context'] = surrounding
            content_image_list.append(c)

    return content_image_list
