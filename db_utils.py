#!/usr/bin/env python3.10
import sqlite3
import asyncio
import aiosqlite
import threading
import logging
import json
import time
import os
from functools import wraps
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Dict, List, Any, Callable, TypeVar, Awaitable

# Configure logging
logger = logging.getLogger(__name__)

# Database configuration - use the same DB_PATH as sikum.py
DB_PATH = os.getenv('DB_PATH', 'bot_stats.db')
MAX_CONNECTIONS = 20  # Limit max connections to avoid resource exhaustion
CACHE_SIZE = 100  # Maximum number of entries in cache
CACHE_TTL = 300   # Maximum time in cache (seconds)

# Connection semaphore for limiting concurrent connections
connection_semaphore = asyncio.Semaphore(MAX_CONNECTIONS)
local = threading.local()  # For thread-local storage of connections

# Function decorator for resilient DB operations with retry and timeout
def resilient_db_operation(func=None, max_retries=3, retry_delay=0.1, timeout=5.0, fallback=None):
    """
    Decorator for making database operations more resilient with retry logic and timeouts.
    
    Args:
        func: The function to decorate
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries (seconds)
        timeout: Maximum time to wait for operation (seconds)
        fallback: Function to call if all retries fail
    
    Returns:
        Decorated function with retry and timeout logic
    """
    def decorator(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    # Create a task and wait with timeout
                    task = asyncio.create_task(f(*args, **kwargs))
                    return await asyncio.wait_for(task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Database operation timed out after {timeout}s")
                    last_exception = asyncio.TimeoutError(f"Operation timed out after {timeout}s")
                    # No retry on timeout, break immediately
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "busy" in str(e):
                        logger.warning(f"Database locked/busy, retrying ({attempt+1}/{max_retries})")
                        last_exception = e
                        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        # For other operational errors, log and retry
                        logger.error(f"Database error: {e}, retrying ({attempt+1}/{max_retries})")
                        last_exception = e
                        await asyncio.sleep(retry_delay)
                except Exception as e:
                    logger.error(f"Unexpected error in database operation: {e}")
                    last_exception = e
                    break  # Don't retry on unexpected errors
            
            # If we get here, all retries failed
            if fallback is not None:
                logger.warning("All database retries failed, using fallback")
                return fallback()
            else:
                logger.error("All database retries failed, no fallback available")
                raise last_exception or Exception("Database operation failed after retries")
        
        return wrapper
    
    if func is None:
        return decorator
    return decorator(func)

# Synchronous connection management
@contextmanager
def get_db_connection():
    """Get a thread-local database connection with optimized settings."""
    if not hasattr(local, 'connection'):
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        
        # Apply optimized settings
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=30000000")
        
        local.connection = conn
    
    try:
        yield local.connection
    except Exception as e:
        logger.error(f"Error in database connection: {e}")
        raise

# Async connection pool management
@asynccontextmanager
async def get_async_db_connection():
    """
    Get an async database connection from the pool with optimized settings.
    Uses a semaphore to limit the number of concurrent connections.
    """
    async with connection_semaphore:
        conn = None
        try:
            conn = await aiosqlite.connect(DB_PATH)
            conn.row_factory = aiosqlite.Row
            
            # Apply optimized settings
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA mmap_size=30000000")
            
            yield conn
            
            # Auto-commit before closing
            await conn.commit()
        except Exception as e:
            logger.error(f"Error in async DB connection: {e}")
            raise
        finally:
            if conn:
                try:
                    await conn.close()
                except Exception:
                    pass  # Ignore errors on close

# A simple cache implementation for database query results
class QueryCache:
    """A simple time-based cache to reduce database load."""
    
    def __init__(self, max_size=CACHE_SIZE, ttl=CACHE_TTL):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get item from cache if it exists and isn't expired."""
        if key in self.cache:
            entry = self.cache[key]
            # Check if entry is expired
            if time.time() - entry['timestamp'] > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry['data']
        self.misses += 1
        return None
    
    def set(self, key, data):
        """Store item in cache with timestamp."""
        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.items(), key=lambda x: x[1]['timestamp'])[0]
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def invalidate(self, key=None):
        """Invalidate specific key or entire cache."""
        if key is None:
            self.cache.clear()
        elif key in self.cache:
            del self.cache[key]
    
    def get_stats(self):
        """Return cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }

# Initialize global query cache
query_cache = QueryCache()

# Database optimization functions
@resilient_db_operation(max_retries=5, timeout=10.0)
async def optimize_db():
    """Optimize database for better performance under concurrent load."""
    async with get_async_db_connection() as conn:
        # Run ANALYZE to update statistics
        await conn.execute("ANALYZE")
        
        # Set optimal WAL checkpoint threshold
        await conn.execute("PRAGMA wal_autocheckpoint=1000")
        
        # Run integrity check in case of corruption
        result = await conn.execute("PRAGMA integrity_check")
        integrity = await result.fetchone()
        if integrity[0] != "ok":
            logger.error(f"Database integrity check failed: {integrity[0]}")
        
        logger.info("Database optimized for concurrent use")

@resilient_db_operation(max_retries=2, timeout=60.0)
async def vacuum_db():
    """Vacuum the database to reclaim space and defragment."""
    async with get_async_db_connection() as conn:
        logger.info("Starting database VACUUM operation")
        await conn.execute("VACUUM")
        logger.info("Database VACUUM completed")

async def init_db():
    """Initialize database with optimized settings for concurrent access."""
    try:
        async with get_async_db_connection() as conn:
            # Create tables with optimized settings
            await conn.executescript('''
                CREATE TABLE IF NOT EXISTS saved_quizzes (
                    quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    questions_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    times_played INTEGER DEFAULT 0,
                    share_token TEXT UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS user_activity (
                    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    action_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS quiz_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    quiz_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    report_reason TEXT NOT NULL,
                    reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (quiz_id) REFERENCES saved_quizzes (quiz_id)
                );
                
                -- Create subscriptions table
                CREATE TABLE IF NOT EXISTS subscriptions (
                    user_id INTEGER PRIMARY KEY,
                    subscribed_until DATE
                );
                
                -- Create daily_usage table
                CREATE TABLE IF NOT EXISTS daily_usage (
                    user_id INTEGER,
                    date DATE,
                    attempts INTEGER DEFAULT 1,
                    PRIMARY KEY (user_id, date)
                );
            ''')
            
            # Create indexes for faster lookups
            await conn.executescript('''
                CREATE INDEX IF NOT EXISTS idx_user_quizzes ON saved_quizzes (user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_share_token ON saved_quizzes (share_token);
                CREATE INDEX IF NOT EXISTS idx_user_activity ON user_activity (user_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_quiz_reports ON quiz_reports (quiz_id, status);
                CREATE INDEX IF NOT EXISTS idx_subscriptions_expiry ON subscriptions (subscribed_until);
                CREATE INDEX IF NOT EXISTS idx_daily_usage_date ON daily_usage (date);
            ''')
            
            logger.info("Database initialized with optimized settings")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

# Helper functions for working with cached quiz data
async def get_quiz_with_caching(quiz_id):
    """Get quiz with caching for better performance."""
    cache_key = f"quiz_{quiz_id}"
    
    # Try cache first
    cached_quiz = query_cache.get(cache_key)
    if cached_quiz:
        return cached_quiz
    
    # Cache miss, get from database
    @resilient_db_operation(fallback=lambda: None)
    async def db_operation():
        async with get_async_db_connection() as conn:
            cursor = await conn.execute(
                'SELECT questions_json FROM saved_quizzes WHERE quiz_id = ?', 
                (quiz_id,)
            )
            result = await cursor.fetchone()
            if result:
                data = json.loads(result[0])
                # Store in cache for future requests
                query_cache.set(cache_key, data)
                return data
            return None
    
    # Use the decorated function directly
    return await db_operation()

async def get_user_quizzes_with_caching(user_id, limit=10):
    """Get user's quizzes with caching for list views."""
    # This is a composite key in our cache
    cache_key = f"user_{user_id}_quizzes"
    cached_data = query_cache.get(cache_key)
    if cached_data:
        return cached_data
    
    # Cache miss, get from database
    @resilient_db_operation(fallback=lambda: [])
    async def db_operation():
        async with get_async_db_connection() as conn:
            cursor = await conn.execute('''
                SELECT quiz_id, title, created_at, times_played
                FROM saved_quizzes
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            # Convert to list of dicts for easier usage
            rows = await cursor.fetchall()
            result = [{
                'quiz_id': row[0],
                'title': row[1],
                'created_at': row[2],
                'times_played': row[3]
            } for row in rows]
            
            # Cache the result
            if result:
                query_cache.set(cache_key, result)
            
            return result
    
    # Use the decorated function directly
    return await db_operation()

# Schedule periodic database maintenance
async def schedule_db_maintenance():
    """Schedule periodic database maintenance tasks."""
    while True:
        try:
            # Run once per day (86400 seconds)
            await asyncio.sleep(86400)
            
            # Run optimization
            await optimize_db()
            
            # Run vacuum if needed (once a week)
            from datetime import datetime
            day_of_week = datetime.now().weekday()
            if day_of_week == 6:  # Sunday
                await vacuum_db()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in scheduled maintenance: {e}")
            await asyncio.sleep(3600)  # Retry in an hour

# Start the maintenance task
def start_maintenance_task():
    """Start the database maintenance task in the background."""
    loop = asyncio.get_event_loop()
    maintenance_task = loop.create_task(schedule_db_maintenance())
    return maintenance_task 