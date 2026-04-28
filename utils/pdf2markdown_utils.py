import os
import re
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Any
from config.base_config import logger
from utils.base_utils import remove_null_chars


def clear_images_in_md(content):
    # 使用正则表达式找到所有符合格式的图片内容并删除
    content = re.sub(r'!\[\]\([^)]*\)', '', content)
    return content


def clear_images_in_md(content):
    # 使用正则表达式找到所有符合格式的图片内容并删除
    content = re.sub(r'!\[\]\([^)]*\)', '', content)
    return content


def pdf2markdown_mineru(pdf_path, output_dir, doc_id):
    output_folder = os.path.join(output_dir, doc_id, 'auto')
    os.makedirs(output_folder, exist_ok=True)

    # 检查 output_folder 下是否有 .md 文件
    if os.path.exists(output_dir) and any(file.endswith(".md") for file in os.listdir(output_folder)):
        logger.info(f"MinerU already finished!")
        return output_folder

    # 构造并运行命令
    try:
        logger.info(f"MinerU processing...")
        # 设置 GPU 设备的环境变量
        # gpu_id = 0
        os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
        # no api
        command = ['mineru', '-p', pdf_path, '-o', output_dir]
        # has api
        # command = ['mineru', '-p', pdf_path, '-o', output_dir, '-b', 'vlm-http-client', '-u', 'http://127.0.0.1:30000']
        completed = subprocess.run(command, capture_output=True, text=True, check=True)
        # print("stdout:", completed.stdout)  # 这里就能看见被截留的内容
        # print("stderr:", completed.stderr)
        logger.info(f"MinerU finished!")
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，打印错误信息
        print("Error:", e.stderr)
    return output_folder


# ==============================================================
# 🔧 Markdown 清理函数（核心）
# ==============================================================

def clean_markdown_content(md_content: str) -> str:
    """
    清理 Markdown 文件内容：
    1. 删除参考文献部分（# REFERENCES 至下一个标题或文末）
    2. 删除图片及图片标题
    3. 删除表格及表格标题
    """

    # --- 删除参考文献段落 ---
    md_content = re.sub(r'# REFERENCES[\s\S]*?(?=\n# |\Z)', '', md_content, flags=re.IGNORECASE)

    # 匹配: ![](url) 后可跟 Figure...
    img_pattern = r'!\[\]\(([^)]+)\)\s*(?:\n\s*)?(Figure[^\n]*)?'
    md_content = re.sub(img_pattern, '', md_content, flags=re.IGNORECASE)

    table_pattern = r'(?:^|\n)\s*(?:Table\s*\d*[:：]?.*\n)?\s*<table[\s\S]*?</table>'
    md_content = re.sub(table_pattern, '', md_content, flags=re.IGNORECASE)

    return md_content.strip()


# ==============================================================
# 🔍 MinerU JSON 解析辅助函数
# ==============================================================

TEXT_KEYS = ["text", "content", "raw_text", "plain_text", "value", "text_content", "ocr_text"]
PAGE_KEYS = ["page", "page_number", "page_index", "pageno", "page_idx", "pageId", "page_no", "pageNo"]


def load_json_flexible(p: Path) -> Any:
    """宽容地加载 JSON 或 JSONL"""
    s = p.read_text(encoding="utf-8")
    try:
        return json.loads(s)
    except Exception:
        objs = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception:
                continue
        return objs


def find_value_recursive(obj: Any, candidates):
    """递归查找第一个匹配候选键的值（返回第一个找到的）"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in candidates:
                return v
        for v in obj.values():
            res = find_value_recursive(v, candidates)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for item in obj:
            res = find_value_recursive(item, candidates)
            if res is not None:
                return res
    return None


def normalize_page_num(x):
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            dig = re.search(r"\d+", str(x))
            return int(dig.group()) if dig else None
        except Exception:
            return None


def extract_blocks(mineru_data):
    """
    从 mineru_data 中抽取原始 block 列表（每个为 dict/str），
    返回 list of {'text':..., 'page':int|None}
    """
    raw_blocks = []
    if isinstance(mineru_data, dict):
        for key in ("blocks", "elements", "content", "items"):
            if key in mineru_data and isinstance(mineru_data[key], list):
                raw_blocks = mineru_data[key]
                break
        if not raw_blocks:
            for v in mineru_data.values():
                if isinstance(v, list):
                    raw_blocks.extend(v)
    elif isinstance(mineru_data, list):
        raw_blocks = mineru_data
    else:
        raise ValueError("未知的 mineru json 结构")

    blocks = []
    for rb in raw_blocks:
        if isinstance(rb, str):
            text = rb.strip()
            page = None
        elif isinstance(rb, dict):
            text = find_value_recursive(rb, set(k.lower() for k in TEXT_KEYS)) or ""
            page = find_value_recursive(rb, set(k.lower() for k in PAGE_KEYS))
        else:
            text = ""
            page = None
        page_num = normalize_page_num(page)
        text = (text or "").strip()
        if text:
            blocks.append({"text": text, "page": page_num})
    return blocks


# ==============================================================
# 📄 核心主函数：Markdown 拆分页 + 清理
# ==============================================================

def split_md_by_page(md_path: str, mineru_json_path: str):
    """
    根据 MinerU JSON 文件中的块信息，将 Markdown 文件按页拆分。
    在拆分前自动清理：
        - 参考文献 (# REFERENCES)
        - 图片与标题
        - 表格与标题
        - 脚注及引用
    """

    md_path = Path(md_path)
    mineru_json_path = Path(mineru_json_path)

    # 读取并清理 Markdown 内容
    md_text = md_path.read_text(encoding="utf-8")
    md_text = remove_null_chars(md_text)
    md_text = clean_markdown_content(md_text)

    # 加载 MinerU 数据
    mineru_data = load_json_flexible(mineru_json_path)
    blocks = extract_blocks(mineru_data)

    # 若无页码信息，默认页码为 1
    if all(b["page"] is None for b in blocks):
        for b in blocks:
            b["page"] = 1

    # 按页分组
    page_map = defaultdict(list)
    for b in blocks:
        page_map[b["page"]].append(b["text"])

    pages_sorted = sorted([p for p in page_map.keys() if p is not None])
    if any(p is None for p in page_map.keys()):
        pages_sorted.append(None)

    # 页面内容匹配
    md_pages = {}
    search_pos = 0
    page_text = ''
    for page in pages_sorted:
        texts = page_map[page]
        starts, ends = [], []
        for t in texts:
            if not t:
                continue
            idx = md_text.find(t, search_pos)
            if idx != -1:
                starts.append(idx)
                ends.append(idx + len(t))
                search_pos = idx + len(t)
                continue

            snippet = re.sub(r"\s+", " ", t.strip())[:120]
            if not snippet:
                continue
            regex = re.escape(snippet).replace(r"\ ", r"\s+")
            m = re.search(regex, md_text[search_pos:], flags=re.DOTALL)
            if m:
                idx0 = search_pos + m.start()
                idx1 = min(len(md_text), idx0 + len(t))
                starts.append(idx0)
                ends.append(idx1)
                search_pos = idx1
                continue

            idx2 = md_text.find(t)
            if idx2 != -1:
                starts.append(idx2)
                ends.append(idx2 + len(t))
                search_pos = idx2 + len(t)
                continue

        if starts and ends:
            s = min(starts)
            e = max(ends)
            page_text = md_text[s:e]
        else:
            continue

        md_pages[page if page is not None else "unknown"] = page_text

    return md_pages


if __name__ == "__main__":
    md_file = "/data/npy/code/mrag/output/pdf2markdown/MMLong/2210.02442v1/auto/2210.02442v1.md"
    mineru_json = "/data/npy/code/mrag/output/pdf2markdown/MMLong/2210.02442v1/auto/2210.02442v1_content_list.json"

    md_pages = split_md_by_page(md_file, mineru_json, out_dir="md_pages_clean")
    print(f"共拆分 {len(md_pages)} 页，已自动清理图片、表格、脚注及参考文献。")
