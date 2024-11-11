LOG=False

def log(*args, **kwargs):
    bypass = kwargs.get("bypass")
    if LOG or bypass:
        print(*args, **kwargs)