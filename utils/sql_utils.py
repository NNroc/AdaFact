import redis
import functools
import json
import hashlib

# 建立 Redis 连接 (保持不变)
try:
    REDIS_CLIENT = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    REDIS_CLIENT.ping()
except redis.exceptions.ConnectionError as e:
    REDIS_CLIENT = None


def redis_cache(expire=None):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if REDIS_CLIENT is None:
                print("⚠️ Redis 不可用，直接执行原函数。")
                return func(*args, **kwargs)

            params_tuple = (args, tuple(sorted(kwargs.items())))
            try:
                params_str = json.dumps(params_tuple)
            except TypeError:
                params_str = repr(params_tuple)

            key_hash = hashlib.sha256(params_str.encode('utf-8')).hexdigest()[:16]
            cache_key = f"{func.__name__}:{key_hash}"

            cached_result = REDIS_CLIENT.get(cache_key)

            if cached_result:
                try:
                    return cached_result
                except Exception:
                    print(f"❌ 缓存数据解析失败，执行原函数。")

            result = func(*args, **kwargs)

            try:
                if expire is None or expire <= 0:
                    REDIS_CLIENT.set(cache_key, result)
                else:
                    # 使用 SETEX 命令，设置过期时间
                    REDIS_CLIENT.setex(cache_key, expire, result)

            except Exception as e:
                print(f"❌ 结果无法序列化或存储失败，不进行缓存。错误: {e}")

            return result

        return wrapper

    return decorator