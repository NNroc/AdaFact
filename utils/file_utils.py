import os
import base64
import json
from io import BytesIO
from PIL import Image

from utils.sql_utils import redis_cache


# 将json对象写入文件
def write_json(json_obj, file_name):
    directory = os.path.dirname(file_name)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


# 从文件中加载json对象
def load_json(file_name):
    try:
        with open(file_name, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"{e},file_name={file_name}")
        return None


# 从文件中加载
def load_file(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()

@redis_cache()
def load_base64(file_name, max_side_pixels=1536):
    with open(file_name, "rb") as image_file:
        img = Image.open(image_file).convert("RGB")
        width, height = img.size
        reduce = 1.0
        if max(width, height) > max_side_pixels:
            reduce = max(width, height) / max_side_pixels
        if reduce > 1.0:
            # 长宽各缩小reduce倍
            new_size = (int(img.width // reduce), int(img.height // reduce))
            # 确保最小尺寸为1
            new_size = (max(1, new_size[0]), max(1, new_size[1]))
            img = img.resize(new_size)
            # 写入内存并编码
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            img.close()
            return base64.b64encode(buffer.read()).decode("utf-8")
        image_file.seek(0)
        img.close()
        return base64.b64encode(image_file.read()).decode("utf-8")

def prepare_files(root_dir: str, suffix=".pdf"):
    """Prepare the list of files in the `root_dir` with the specified `suffix`"""
    target_files = [file for file in os.listdir(root_dir) if file.endswith(suffix)]
    return target_files
