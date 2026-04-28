import argparse
def get_shared_parser():
    """定义可复用的模型与环境参数"""
    parser = argparse.ArgumentParser(add_help=False)
    # "MMLong", "LongDocURL", "PaperTab", "FetaTab"
    parser.add_argument("--dataset", type=str, default="MMLong")
    parser.add_argument('--dataset_dir', type=str, default='/data/npy/datasets/VLMRAG')
    parser.add_argument('--databases_save_dir', type=str, default='/home/npy/output/databases')
    parser.add_argument('--results_save_dir', type=str, default='/home/npy/output/results')
    parser.add_argument('--minure_save_dir', type=str, default='/home/npy/output/pdf2markdown')
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--TEXT_ENCODER_MODEL", type=str, default="Qwen3-Embedding-4B")
    parser.add_argument("--MM_ENCODER_MODEL", type=str, default="colqwen2.5-v0.2")
    parser.add_argument("--API_KEY", type=str, default="API_KEY")
    parser.add_argument("--MM_API_KEY", type=str, default="MM_API_KEY")
    parser.add_argument("--MODEL", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--URL", type=str, default="")
    parser.add_argument("--MM_MODEL", type=str, default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--MM_URL", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument('--debug', action='store_true', help="only use debug")
    return parser
