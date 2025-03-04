#!/usr/bin/env python3
import sqlite3
import os

def init_db():
    """Initialize the bot_stats.db file with all required tables."""
    db_path = os.getenv('DB_PATH', 'bot_stats.db')
    print(f"Initializing database at {db_path}")
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create all necessary tables
    c.executescript('''
        -- Main quiz storage table
        CREATE TABLE IF NOT EXISTS saved_quizzes (
            quiz_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            questions_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            times_played INTEGER DEFAULT 0,
            share_token TEXT UNIQUE
        );
        
        -- User activity tracking
        CREATE TABLE IF NOT EXISTS user_activity (
            activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action_type TEXT NOT NULL,
            action_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Quiz reporting system
        CREATE TABLE IF NOT EXISTS quiz_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            report_reason TEXT NOT NULL,
            reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (quiz_id) REFERENCES saved_quizzes (quiz_id)
        );
        
        -- Subscription tracking
        CREATE TABLE IF NOT EXISTS subscriptions (
            user_id INTEGER PRIMARY KEY,
            subscribed_until DATE
        );
        
        -- Daily usage tracking
        CREATE TABLE IF NOT EXISTS daily_usage (
            user_id INTEGER,
            date DATE,
            attempts INTEGER DEFAULT 1,
            PRIMARY KEY (user_id, date)
        );
    ''')
    
    # Create indexes for better performance
    c.executescript('''
        CREATE INDEX IF NOT EXISTS idx_user_quizzes ON saved_quizzes (user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_share_token ON saved_quizzes (share_token);
        CREATE INDEX IF NOT EXISTS idx_user_activity ON user_activity (user_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_quiz_reports ON quiz_reports (quiz_id, status);
        CREATE INDEX IF NOT EXISTS idx_subscriptions_expiry ON subscriptions (subscribed_until);
        CREATE INDEX IF NOT EXISTS idx_daily_usage_date ON daily_usage (date);
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

if __name__ == "__main__":
    init_db() 