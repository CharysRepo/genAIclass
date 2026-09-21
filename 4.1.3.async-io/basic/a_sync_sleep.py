import time

def sleep_for_some_time(value: int) -> int:
    time.sleep(1)
    return value * 10


def main() -> None:
    tot_count = 0
    for i in range(100):
        tot_count += sleep_for_some_time(i)
    print(tot_count)


start_time = time.perf_counter()
main()
end_time = time.perf_counter()
print(f"Time taken(Sync): {end_time - start_time:.2f} seconds")
