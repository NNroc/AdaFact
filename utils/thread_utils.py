import threading
import asyncio


def run_async_in_thread(coroutine):
    """在独立线程中运行异步代码"""
    result = None
    event = threading.Event()

    def run():
        nonlocal result
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coroutine)
        except Exception as e:
            result = e
        finally:
            loop.close()
            event.set()

    thread = threading.Thread(target=run)
    thread.start()
    event.wait()  # 等待完成
    if isinstance(result, Exception):
        raise result
    return result
