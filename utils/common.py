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

def load_cookies():
    with open("linkedin_cookies.json", "r") as f:
        return json.load(f)

def convert_seconds_to_readable(seconds: int) -> str:
    hours = round(seconds // 3600)
    minutes = round((seconds % 3600) // 60)
    seconds = round(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"