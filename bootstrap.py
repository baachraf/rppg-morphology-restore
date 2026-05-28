"""
Session entry point. Run at the start of every session.
python bootstrap.py
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB   = Path(__file__).parent / "research.db"
GOALS = Path(__file__).parent / "PROJECT_GOALS.md"
BLOC  = Path(__file__).parent / "blockers.txt"
PROJECT = Path(__file__).parent.name

MLFLOW_URI = "http://localhost:5000"
SEP = "=" * 60


def read_goals_summary():
    if not GOALS.exists():
        return "PROJECT_GOALS.md not found"
    lines = GOALS.read_text(encoding="utf-8").splitlines()
    for i, l in enumerate(lines):
        if l.startswith("## Scientific Question"):
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip():
                    return lines[j].strip()
    return "See PROJECT_GOALS.md"


def read_blockers():
    if not BLOC.exists():
        return []
    return [l.strip() for l in BLOC.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def main():
    print(f"\n{SEP}")
    print(f"  PROJECT: {PROJECT} — {datetime.now().strftime('%Y-%m-%d')}")
    print(SEP)
    print(f"  GOAL: {read_goals_summary()}")
    print()

    if not DB.exists():
        print("  [!] research.db not found — initializing...")
        import init_db
        init_db.init()
        print("  [OK] research.db created.\n")

    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row

    # Valid results
    valid = con.execute("""
        SELECT arch, metrics, hypothesis, interpretation, implication, timestamp
        FROM experiments WHERE status='valid'
        ORDER BY timestamp DESC
    """).fetchall()

    if valid:
        print(f"  VALID RESULTS ({len(valid)}):")
        for r in valid:
            m = json.loads(r["metrics"]) if r["metrics"] else {}
            metrics_str = "  ".join(f"{k}={v}" for k, v in m.items())
            print(f"    {r['arch']:<20s}  {metrics_str}")
            if r["interpretation"]:
                print(f"      interprets : {r['interpretation'][:80]}")
            if r["implication"]:
                print(f"      implies    : {r['implication'][:80]}")
        print()

    # Invalid
    invalid = con.execute("""
        SELECT arch, invalid_reason, interpretation FROM experiments
        WHERE status='invalid' ORDER BY timestamp DESC
    """).fetchall()
    if invalid:
        print(f"  INVALID ({len(invalid)}):")
        for r in invalid:
            print(f"    {r['arch']:<20s}  reason={r['invalid_reason']}")
        print()

    # Pending
    pending = con.execute("""
        SELECT arch, hypothesis, timestamp FROM experiments
        WHERE status='pending' ORDER BY timestamp DESC
    """).fetchall()
    if pending:
        print(f"  PENDING ({len(pending)}):")
        for r in pending:
            print(f"    {r['arch']:<20s}  hypothesis logged")
        print()

    if not valid and not invalid and not pending:
        print("  No experiments registered yet.\n")

    # Blockers
    blockers = read_blockers()
    if blockers:
        print("  BLOCKERS:")
        for b in blockers:
            print(f"    {b}")
        print()

    con.close()
    print(f"  MLflow UI : {MLFLOW_URI}")
    print(f"  DB        : {DB}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
