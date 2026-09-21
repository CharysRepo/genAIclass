import asyncio
import threading
import time


async def sleep_for_some_time(value: int) -> int:
    print("tid:%s" % threading.get_ident())
    await asyncio.sleep(1)
    return value * 10


async def main() -> None:
    tot_count = 0
    for i in range(10):
        res = await sleep_for_some_time(i)
        tot_count += res
    print(tot_count)


start_time = time.perf_counter()
asyncio.run(main())
end_time = time.perf_counter()
print(f"Time taken(Async-no batch): {end_time - start_time:.2f} seconds")
