import asyncio
import time
from asyncio import events
import threading


async def sleep_for_some_time(value: int) -> int:
    print("tid:%s" % threading.get_ident())
    await asyncio.sleep(1)
    return value * 10


async def main() -> None:
    tasks = [sleep_for_some_time(i) for i in range(10)]
    values = await asyncio.gather(*tasks)
    print(sum(values))


def run(fn):
    loop = events.new_event_loop()
    loop.run_until_complete(fn)
    loop.close()


start_time = time.perf_counter()
run(main())
end_time = time.perf_counter()
print(f"Time taken(Async-batch): {end_time - start_time:.2f} seconds")
