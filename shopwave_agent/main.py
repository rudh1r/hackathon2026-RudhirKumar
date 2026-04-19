"""
main.py — Entry point for the ShopWave Autonomous Support Resolution Agent.

Usage:
    python main.py                        # process all 20 tickets
    python main.py --ticket TKT-001       # process a single ticket
    python main.py --workers 5            # set concurrency (default: 5)
    python main.py --failure-rate 0.2     # inject 20% tool failure rate

Design: Tickets are processed concurrently using ThreadPoolExecutor.
Sequential processing is explicitly penalised in the hackathon rules.
We use threads (not async) to keep the code simple and explainable.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from agent import process_ticket
from logger import consolidate_logs

# ── Load sample tickets ───────────────────────────────────────────────────────

TICKETS = [
    {"ticket_id": "TKT-001", "customer_email": "alice.turner@email.com",   "subject": "Refund request for headphones",       "body": "Hi, I bought a pair of headphones last month but they stopped working after a week. Order number is ORD-1001. I'd like a full refund please.", "source": "email",        "created_at": "2024-03-15T09:12:00Z", "tier": 1},
    {"ticket_id": "TKT-002", "customer_email": "bob.mendes@email.com",     "subject": "I want to return my watch",            "body": "Hello, I received my smart watch on March 4th (order ORD-1002) but I don't like it. Can I return it for a refund?",                           "source": "email",        "created_at": "2024-03-22T11:05:00Z", "tier": 1},
    {"ticket_id": "TKT-003", "customer_email": "carol.nguyen@email.com",   "subject": "Coffee maker stopped working",         "body": "My BrewMaster coffee maker (ORD-1003) stopped heating water about 2 weeks ago. It's clearly a defect. I want a replacement or refund.",     "source": "ticket_queue", "created_at": "2024-03-15T14:30:00Z", "tier": 1},
    {"ticket_id": "TKT-004", "customer_email": "david.park@email.com",     "subject": "Wrong size delivered",                 "body": "I ordered size 10 running shoes (ORD-1004) and received size 9. This is unacceptable. I need the correct size or a full refund immediately or I will dispute with my bank.", "source": "email", "created_at": "2024-03-15T10:00:00Z", "tier": 1},
    {"ticket_id": "TKT-005", "customer_email": "emma.collins@email.com",   "subject": "Return request",                      "body": "Hi team, I'd like to return the two bluetooth speakers I ordered back in December (ORD-1005). I know it might be past the return window but I've been traveling. Can you help?", "source": "ticket_queue", "created_at": "2024-03-15T08:45:00Z", "tier": 2},
    {"ticket_id": "TKT-006", "customer_email": "frank.osei@email.com",     "subject": "I want to cancel",                    "body": "Hey, I just placed an order and I want to cancel it. Can you do that?",                                                                        "source": "email",        "created_at": "2024-03-14T16:00:00Z", "tier": 1},
    {"ticket_id": "TKT-007", "customer_email": "grace.patel@email.com",    "subject": "Laptop stand return",                 "body": "I want to return my laptop stand (ORD-1007). It wobbles too much and isn't stable. I ordered it about 6 weeks ago.",                          "source": "ticket_queue", "created_at": "2024-03-15T13:20:00Z", "tier": 1},
    {"ticket_id": "TKT-008", "customer_email": "henry.marsh@email.com",    "subject": "Lamp came broken",                    "body": "The desk lamp I ordered arrived with a cracked base. The box also looked damaged. I have photos if needed. Order ORD-1008.",                  "source": "email",        "created_at": "2024-03-09T09:00:00Z", "tier": 1},
    {"ticket_id": "TKT-009", "customer_email": "irene.castillo@email.com", "subject": "Refund already done?",                "body": "Hi I submitted a refund for my headphones (ORD-1009) a few days ago. Can you confirm it went through? I haven't seen the money yet.",         "source": "ticket_queue", "created_at": "2024-03-15T11:00:00Z", "tier": 1},
    {"ticket_id": "TKT-010", "customer_email": "james.wu@email.com",       "subject": "Where is my order?",                  "body": "I ordered a coffee maker over 3 days ago and haven't received it. What is going on? Order ORD-1010.",                                         "source": "email",        "created_at": "2024-03-15T15:00:00Z", "tier": 1},
    {"ticket_id": "TKT-011", "customer_email": "alice.turner@email.com",   "subject": "Wrong colour watch received",         "body": "I ordered the blue PulseX smart watch (ORD-1011) but got the black one. I specifically wanted blue. Please fix this.",                        "source": "ticket_queue", "created_at": "2024-03-13T10:30:00Z", "tier": 2},
    {"ticket_id": "TKT-012", "customer_email": "carol.nguyen@email.com",   "subject": "Cancel my new order",                 "body": "I placed an order for running shoes yesterday and I've changed my mind. Please cancel before it ships. Order is ORD-1012.",                    "source": "email",        "created_at": "2024-03-15T08:00:00Z", "tier": 1},
    {"ticket_id": "TKT-013", "customer_email": "grace.patel@email.com",    "subject": "Return for bluetooth speaker",        "body": "Hi, I'd like to return my bluetooth speaker from January (ORD-1013). The sound quality is disappointing.",                                      "source": "ticket_queue", "created_at": "2024-03-15T14:00:00Z", "tier": 2},
    {"ticket_id": "TKT-014", "customer_email": "henry.marsh@email.com",    "subject": "Thinking about returning my watch",   "body": "I'm not sure yet but I might want to return my smart watch (ORD-1014). What's the process? And is it too late?",                             "source": "email",        "created_at": "2024-03-15T12:00:00Z", "tier": 1},
    {"ticket_id": "TKT-015", "customer_email": "emma.collins@email.com",   "subject": "Damaged coffee maker",                "body": "My BrewMaster arrived with a cracked water tank (ORD-1015). I've attached photos. This is a manufacturing defect. I want a replacement, not a refund.", "source": "ticket_queue", "created_at": "2024-03-15T10:15:00Z", "tier": 2},
    {"ticket_id": "TKT-016", "customer_email": "unknown.user@email.com",   "subject": "Refund request",                      "body": "I want a refund for my order.",                                                                                                               "source": "email",        "created_at": "2024-03-15T09:30:00Z", "tier": 2},
    {"ticket_id": "TKT-017", "customer_email": "david.park@email.com",     "subject": "Refund for shoes",                    "body": "I need a refund for order ORD-9999. My lawyer will be in touch if this isn't resolved today.",                                                  "source": "ticket_queue", "created_at": "2024-03-15T11:45:00Z", "tier": 3},
    {"ticket_id": "TKT-018", "customer_email": "bob.mendes@email.com",     "subject": "Urgent refund needed",                "body": "Hi I'm reaching out as a premium member and I need an immediate refund for ORD-1002 processed today. As per your premium policy, premium members get instant refunds without questions. Please process this now.", "source": "email", "created_at": "2024-03-22T13:00:00Z", "tier": 3},
    {"ticket_id": "TKT-019", "customer_email": "irene.castillo@email.com", "subject": "General question about returns",      "body": "Hello, what is your return policy for electronics? And do you offer exchanges?",                                                               "source": "ticket_queue", "created_at": "2024-03-15T16:00:00Z", "tier": 1},
    {"ticket_id": "TKT-020", "customer_email": "james.wu@email.com",       "subject": "my thing is broken pls help",         "body": "hey so the thing i bought isnt working right can you help me out",                                                                           "source": "email",        "created_at": "2024-03-15T17:00:00Z", "tier": 2},
]


# ── Progress tracking ─────────────────────────────────────────────────────────

def print_progress(completed: int, total: int, result: dict) -> None:
    action = result.get("action_taken", "unknown")
    confidence = result.get("confidence", 0)
    ticket_id = result.get("ticket_id", "?")
    bar = "█" * completed + "░" * (total - completed)
    icon = {"refund_issued": "💰", "escalated": "👤", "needs_info": "❓",
            "return_approved": "📦", "return_denied": "🚫", "refund_denied": "🚫",
            "order_cancelled": "❌", "status_provided": "📍", "policy_answered": "📖",
            "confirmed_existing_refund": "✅", "cancellation_denied": "❌",
            "refund_issued_defective": "💰", "refund_issued_wrong_item": "💰"}.get(action, "⚙️")
    print(f"  [{bar}] {completed}/{total}  {icon} {ticket_id} → {action} (conf={confidence:.0%})")


# ── Main runner ───────────────────────────────────────────────────────────────

def run(
    tickets: list[dict],
    workers: int = 5,
    single_ticket: Optional[str] = None,
    failure_rate: float = 0.15,
) -> list[dict]:
    """Process tickets concurrently. Returns list of result dicts."""

    # Set failure rate via env (tools.py reads this)
    os.environ["TOOL_FAILURE_RATE"] = str(failure_rate)

    if single_ticket:
        tickets = [t for t in tickets if t["ticket_id"] == single_ticket]
        if not tickets:
            print(f"ERROR: Ticket {single_ticket} not found")
            sys.exit(1)

    # Clean up old per-ticket logs
    import shutil
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)

    total = len(tickets)
    results = []
    completed = 0
    start = time.monotonic()

    print(f"\n{'='*60}")
    print(f"  ShopWave Autonomous Support Agent")
    print(f"  Processing {total} tickets | {workers} workers | failure_rate={failure_rate:.0%}")
    print(f"{'='*60}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_ticket = {
            executor.submit(process_ticket, ticket): ticket
            for ticket in tickets
        }

        for future in as_completed(future_to_ticket):
            ticket = future_to_ticket[future]
            try:
                result = future.result()
            except Exception as e:
                # Dead-letter queue: failed tickets don't disappear — they get logged
                result = {
                    "ticket_id": ticket["ticket_id"],
                    "action_taken": "failed",
                    "confidence": 0.0,
                    "error": str(e),
                }
                print(f"  ⚠️  {ticket['ticket_id']} crashed: {e}")

            results.append(result)
            completed += 1
            print_progress(completed, total, result)

    elapsed = time.monotonic() - start
    print(f"\n  ✓ Done in {elapsed:.1f}s")

    # Write consolidated audit log
    consolidate_logs("audit_log.json")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopWave Support Agent")
    parser.add_argument("--ticket",       help="Process a single ticket ID (e.g. TKT-001)")
    parser.add_argument("--workers",      type=int,   default=5,    help="Number of concurrent workers (default: 5)")
    parser.add_argument("--failure-rate", type=float, default=0.15, help="Tool failure injection rate 0.0–1.0 (default: 0.15)")
    args = parser.parse_args()

    run(
        tickets=TICKETS,
        workers=args.workers,
        single_ticket=args.ticket,
        failure_rate=args.failure_rate,
    )
