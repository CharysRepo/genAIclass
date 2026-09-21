import time
import concurrent.futures


def sleep_for_some_time(value: int) -> int:
    time.sleep(1)
    return value * 10


def approach3() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(sleep_for_some_time, i) for i in range(100)]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
    print(sum(results))


start_time = time.perf_counter()
approach3()
end_time = time.perf_counter()
print(f"Time taken(Multi threading): {end_time - start_time:.2f} seconds")
