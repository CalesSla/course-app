import asyncio

async def cook(order, time_to_prepare):
    print(f"Getting order {order}")
    await asyncio.sleep(time_to_prepare)
    print(order, "ready")

async def waiter():
    tasks = []
    tasks.append(asyncio.create_task(cook("Beef Steak", 10)))
    tasks.append(asyncio.create_task(cook("Salad", 4)))
    tasks.append(asyncio.create_task(cook("Orange Juice", 2)))

    await asyncio.gather(*tasks)

asyncio.run(waiter())