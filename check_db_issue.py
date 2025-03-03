#!/usr/bin/env python3
import asyncio
from sikum import db_get_user_quizzes_with_caching

async def test_function():
    # Test with a dummy user ID
    result = await db_get_user_quizzes_with_caching(12345)
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_function()) 