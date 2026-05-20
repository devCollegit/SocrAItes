#!/usr/bin/env python3
import sqlite3

db = sqlite3.connect('socraites.db')
cur = db.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print('Tables in database:')
for t in tables:
    print(f'  - {t[0]}')

print('\nChecking schedules table...')
try:
    cur.execute('SELECT COUNT(*) as count FROM schedules')
    count = cur.fetchone()[0]
    print(f'  Schedules: {count} rows')
except Exception as e:
    print(f'  Error: {e}')

print('\nChecking weaknesses table...')
try:
    cur.execute('SELECT COUNT(*) as count FROM weaknesses')
    count = cur.fetchone()[0]
    print(f'  Weaknesses: {count} rows')
except Exception as e:
    print(f'  Error: {e}')

db.close()
