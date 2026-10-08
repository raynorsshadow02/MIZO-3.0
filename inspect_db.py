import sqlite3

conn = sqlite3.connect("data/mizo.db")
conn.row_factory = sqlite3.Row

print("\nSTUDENTS")
print("=" * 80)

rows = conn.execute("""
    SELECT
        id,
        name,
        onboarding_completed,
        onboarding_step,
        profile_version,
        total_sessions,
        created_at,
        updated_at
    FROM students
    ORDER BY id DESC
    LIMIT 30
""").fetchall()

for r in rows:
    print(
        f"ID={r['id']:4} | "
        f"Name={str(r['name']):25} | "
        f"Onboarded={r['onboarding_completed']} | "
        f"Step={str(r['onboarding_step']):15} | "
        f"Sessions={r['total_sessions']} | "
        f"ProfileVer={r['profile_version']} | "
        f"Created={r['created_at']} | "
        f"Updated={r['updated_at']}"
    )

print("\nCONVERSATIONS")
print("=" * 80)

rows = conn.execute("""
    SELECT
        id,
        student_id,
        session_id,
        role,
        substr(content, 1, 100) AS content,
        learning_mode,
        provider_used,
        is_fallback,
        onboarding_state,
        timestamp
    FROM conversations
    ORDER BY id DESC
    LIMIT 30
""").fetchall()

for r in rows:
    print(
        f"ID={r['id']:4} | "
        f"Student={r['student_id']:4} | "
        f"Role={r['role']:8} | "
        f"Session={str(r['session_id'])[:18]:18} | "
        f"Provider={str(r['provider_used']):10} | "
        f"Fallback={r['is_fallback']} | "
        f"State={str(r['onboarding_state']):12} | "
        f"Content={str(r['content'])[:100]}"
    )

print("\nLEARNING SESSIONS")
print("=" * 80)

rows = conn.execute("""
    SELECT
        id,
        session_id,
        student_id,
        mode,
        status,
        message_count,
        created_at,
        updated_at
    FROM learning_sessions
    ORDER BY id DESC
    LIMIT 30
""").fetchall()

for r in rows:
    print(
        f"ID={r['id']:4} | "
        f"Student={r['student_id']:4} | "
        f"Session={str(r['session_id'])[:18]:18} | "
        f"Mode={str(r['mode']):15} | "
        f"Status={str(r['status']):10} | "
        f"Messages={r['message_count']} | "
        f"Created={r['created_at']} | "
        f"Updated={r['updated_at']}"
    )

conn.close()