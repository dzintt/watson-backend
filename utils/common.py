import asyncio
import nest_asyncio

def run_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

def remove_formatting(text: str) -> str:
    return text.replace("`", "").replace("*", "").replace("_", "").replace("\n", " ")