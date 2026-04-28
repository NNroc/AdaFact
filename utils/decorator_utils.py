import functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Iterable
from filelock import FileLock
from tqdm import tqdm

def lazy_property(func):
    attr_name = func.__name__
    # 线程锁：用于防止同一进程内多线程冲突
    _thread_lock = threading.Lock()
    def _lazy_loader(self):
        # 1. 第一层检查：如果已经加载，直接返回
        if attr_name not in self.__dict__:
            # 2. 线程同步：确保同一进程内只有一个线程往下走
            with _thread_lock:
                # 再次检查，防止等锁期间已被其他线程初始化
                if attr_name not in self.__dict__:
                    # 3. 进程同步：使用文件锁，确保多进程间不冲突
                    # 锁文件建议放在模型目录或 /tmp 下
                    lock_path = f"/tmp/{attr_name}_process.lock"
                    with FileLock(lock_path):
                        # 4. 最终检查并执行（针对跨进程的缓存需结合 init_model 里的 LRU）
                        # 注意：init_model 里的 @lru_cache 是进程内私有的
                        # 这里执行真正的初始化逻辑
                        self.__dict__[attr_name] = func(self)

        return self.__dict__[attr_name]

    return property(_lazy_loader)


def singleton_property(func):
    # 用于存储已初始化的实例 (进程内缓存)
    _instances = {}
    # 线程锁，防止多线程同时初始化同一个模型
    _thread_lock = threading.Lock()

    @functools.wraps(func)
    def wrapper(model_name, *args, **kwargs):
        # 1. 第一层检查：如果模型已在内存中，直接返回
        if model_name in _instances:
            return _instances[model_name]

        with _thread_lock:
            # 2. 第二层检查：防止等锁期间被其他线程初始化
            if model_name not in _instances:
                # 3. 进程同步：使用文件锁
                # 将模型名称中的特殊字符替换，避免路径非法
                safe_name = str(model_name).replace("/", "_").replace(".", "_")
                lock_path = f"/tmp/lock_{safe_name}.lock"

                with FileLock(lock_path):
                    # 再次确认缓存
                    if model_name not in _instances:
                        # 执行真正的加载逻辑
                        print(f"[Init] Loading model {model_name} with locks...")
                        _instances[model_name] = func(model_name, *args, **kwargs)

        return _instances[model_name]

    return wrapper

def parallel_processor(mode="thread", max_workers=16, desc: str = 'Processing'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            documents: 待处理的列表
            *args: 透传给 func 的其他位置参数
            **kwargs: 透传给 func 的其他关键字参数
            """
            for arg in args:
                if not isinstance(arg, Iterable):
                    raise ValueError("装饰后的函数参数必须是列迭代器")
            for _, val in kwargs.items():
                if not isinstance(val, Iterable):
                    raise ValueError("装饰后的函数参数必须是迭代器")

            # 根据配置选择执行器
            Executor = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor

            args_iter = [iter(arg) for arg in args]
            kwargs_iter = {k: iter(v) for k, v in kwargs.items()}

            print(f"任务启动 | 模式: {mode} | 并行度: {max_workers}")
            futures = []
            results = []
            with Executor(max_workers=max_workers) as executor:
                while True:
                    try:
                        args = [next(arg_iter) for arg_iter in args_iter]
                        kwargs = {k: next(v) for k, v in kwargs_iter.items()}
                    except StopIteration:
                        break
                    future = executor.submit(func, *args, **kwargs)
                    futures.append(future)

                for future in tqdm(as_completed(futures), total=len(futures), desc=desc, ncols=80):
                    result = future.result()
                    results.append(result)
            return results

        return wrapper

    return decorator

