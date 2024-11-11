import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        elapsed_time = end - start
        if elapsed_time < 1:
            print(f"**Time: {func.__name__} - {(elapsed_time) * 1000:.2f}ms")
        else:
            print(f"**Time: {func.__name__} - {elapsed_time:.2f}s")
    return wrapper

def show_time(_time, name, level=1):
    padding = "\t" * level
    if _time < 1:
        print(f"{padding}**Time: {name} - {(_time) * 1000:.2f}ms")
    else:
        print(f"{padding}**Time: {name} - {_time:.2f}s")

def time_this(func, *args, **kwargs):
    start = time.time()
    output = func(*args, **kwargs)
    end = time.time()
    return output, end - start