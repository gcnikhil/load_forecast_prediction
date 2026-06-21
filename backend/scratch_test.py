import asyncio
from main import get_realtime_status

async def main():
    try:
        res = await get_realtime_status()
        print("SUCCESS:", res)
    except Exception as e:
        print("ERROR:", e)

if __name__ == "__main__":
    asyncio.run(main())
