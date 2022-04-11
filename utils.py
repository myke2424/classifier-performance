import time


def time_it(description=""):
    def decorator(func):
        def inner(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            compute_time = end - start
            print(f"{description} Computational Time: {compute_time}")

        return inner

    return decorator
