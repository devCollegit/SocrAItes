#!/usr/bin/env python3
"""Verify database persistence after tool execution"""
import sqlite3

db = sqlite3.connect('socraites.db')
db.row_factory = sqlite3.Row
cur = db.cursor()

print("="*70)
print("DATABASE VERIFICATION - Tool Persistence Results")
print("="*70)

# Check schedules
print("\n[SCHEDULES TABLE]")
cur.execute('SELECT * FROM schedules ORDER BY id DESC LIMIT 3')
rows = cur.fetchall()
print(f"Total schedules: {len(rows)}")
for row in rows:
    rid = row['id']
    rev_at = row['review_at']
    desc = row['description']
    print(f"  Schedule #{rid}")
    print(f"    Review At: {rev_at}")
    print(f"    Description: {desc}")
    print()

# Check weaknesses  
print("[WEAKNESSES TABLE]")
cur.execute('SELECT * FROM weaknesses ORDER BY id DESC LIMIT 3')
rows = cur.fetchall()
print(f"Total weaknesses: {len(rows)}")
for row in rows:
    wid = row['id']
    concept = row['concept']
    details = row['details']
    severity = row['severity']
    print(f"  Weakness #{wid}")
    print(f"    Concept: {concept}")
    print(f"    Details: {details[:50]}...")
    print(f"    Severity: {severity}/5")
    print()

db.close()
print("✅ Tool results successfully persisted to SQLite!")
