import time


def time_it(description=""):
    """Used to record the computational time of the classifiers"""

    def wrapper(func):
        def inner(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            compute_time = end - start
            print(f"{description} Computational Time: {compute_time}")

        return inner

    return wrapper
