import time
import tracemalloc


def timer(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        f = func(*args, **kwargs)
        end = time.time()
        time_span = (end - begin) * 1000
        # print ('\033[37mTotal time taken:  \033[36m %2.3f ms' %time_span  ,'\033[0m')
        print("Total time taken:  \033[36m %2.3f ms" % time_span, "\033[0m")

        return f

    return wrapper


def mem_measure(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        f = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        # print(f"\n\033[37mMethod Name       :\033[35;1m {func.__name__}\033[0m")
        # print(f"\033[37mCurrent memory usage:\033[36m {current / 10**6}MB\033[0m")
        # print(f"\033[37mPeak                :\033[36m {peak / 10**6}MB\033[0m")
        print(f"Method Name       :\033[35;1m {func.__name__}\033[0m")
        print(f"Current memory usage:\033[36m {current / 10**6}MB\033[0m")
        print(f"Peak                :\033[36m {peak / 10**6}MB\033[0m")
        tracemalloc.stop()
        return f

    return wrapper
