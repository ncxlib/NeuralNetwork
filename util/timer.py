import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(args, kwargs)
        print(f"**Time: {func.__name__} - {(time.time() - start) * 1000:.2f}ms")
    return wrapper