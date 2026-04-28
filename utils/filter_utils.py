import re


def filter_entities(nodes_df):
    """
    过滤实体 DataFrame，移除以下类型：
    1. 包含 'et al' 的内容
    2. 各种数字格式：纯数字、小数、百分比、分数、金钱等

    Args:
        nodes_df: 实体 DataFrame [title, frequency, text_unit_ids]

    Returns:
        filtered_df: 过滤后的 DataFrame
    """

    def should_filter_entity(text):
        """判断实体是否应该被过滤"""
        text = str(text).strip().lower()

        # 过滤条件1: 包含 'et al'
        if 'et al' in text:
            return True

        # 过滤条件2: 各种数字格式
        if is_numeric_format(text):
            return True

        # 过滤条件3: 数学公式
        if is_math_expression(text):
            return True

        return False

    if isinstance(nodes_df, list):
        # 输入是列表，直接过滤列表
        filtered_list = [entity for entity in nodes_df if not should_filter_entity(entity)]
        return filtered_list

    if nodes_df.empty:
        return nodes_df
    mask = ~nodes_df['title'].apply(should_filter_entity)
    filtered_df = nodes_df[mask].reset_index(drop=True)
    return filtered_df


def is_numeric_format(text):
    """检查文本是否为数字格式"""
    numeric_patterns = [
        r'^\d+$',  # 纯数字
        r'^\d+\.\d+$',  # 小数
        r'^\d+/\d+$',  # 分数
        r'^\d+\s*-\s*\d+$',  # 数字范围
        r'^\d*\.?\d+\s*%$',  # 百分比
        r'^\$[\d,]+\.?\d*$',  # 美元
        r'^€[\d,]+\.?\d*$',  # 欧元
        r'^£[\d,]+\.?\d*$',  # 英镑
        r'^¥[\d,]+\.?\d*$',  # 日元
        r'^[\s\d\.]+$',  # 只有空格、数字和小数点
    ]

    return any(re.match(pattern, text) for pattern in numeric_patterns)


def is_math_expression(text):
    """检查文本是否为纯数学表达式（只包含数字、运算符和空格）"""
    # 允许的字符：数字、空格和数学运算符
    # 注意：这里我们使用原始字符串，注意转义
    pattern = r'^[\d\s\+\-\*\/=\.\(\)<>≤≥≈≠]+$'
    # 必须包含至少一个数字，并且整个字符串由允许的字符组成
    return bool(re.match(pattern, text)) and bool(re.search(r'\d', text))
