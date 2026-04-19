# ShopWave Autonomous Support Resolution Agent

> **Hackathon 2026 — Agentic AI Engineering**  
> A production-grade AI agent that autonomously resolves customer support tickets using multi-step tool chains, failure recovery, concurrency, and explainable reasoning.

---

## Quick Start

```bash
# Install dependencies (stdlib only — no external packages required)
python --version  # Python 3.10+ required

# Run all 20 tickets (5 concurrent workers)
python main.py

# Run a single ticket
python main.py --ticket TKT-001

# Run with higher concurrency
python main.py --workers 10

# Increase tool failure rate to stress-test recovery
python main.py --failure-rate 0.3
```

**Output files:**
- `audit_log.json` — Full structured log of all 20 tickets (tool calls, reasoning, decisions)
- `logs/TKT-XXX.json` — Per-ticket audit logs

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     main.py (entry point)                    │
│              ThreadPoolExecutor — N concurrent workers       │
└──────────────────────────┬──────────────────────────────────┘
                           │  one ticket per worker
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     agent.py — ReAct Loop                    │
│                                                              │
│  1. OBSERVE   → Parse ticket, pre-classify intent            │
│  2. THINK     → Plan tool chain strategy                     │
│  3. ACT       → Execute tools (min 3 calls per ticket)       │
│  4. REFLECT   → Self-critique + confidence scoring           │
│       ↓                                                      │
│  Branch by intent (refund / return / cancel / status / etc.) │
│       ↓                                                      │
│  Commit action OR escalate if confidence < 0.6               │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌─────────────────────┐      ┌────────────────────────────┐
│     tools.py        │      │       logger.py             │
│                     │      │                             │
│  READ tools:        │      │  TicketLogger:              │
│  • get_order        │      │  • .observe() .think()      │
│  • get_customer     │      │  • .act() .reflect()        │
│  • get_product      │      │  • .tool_call()             │
│  • search_kb        │      │  • .set_outcome()           │
│                     │      │  • .finalize() → JSON       │
│  WRITE tools:       │      │                             │
│  • check_refund_eli │      │  consolidate_logs()         │
│  • issue_refund     │      │  → audit_log.json           │
│  • send_reply       │      └────────────────────────────┘
│  • escalate         │
│                     │
│  Failure injection: │
│  ToolTimeoutError   │
│  ToolDataError      │
└─────────────────────┘
```

---

## Design Decisions

### 1. ReAct Loop (not just one LLM call)
The agent follows **Observe → Think → Act → Reflect** for every ticket. This makes every decision traceable. There are no one-shot black-box outputs.

### 2. Rule-based pre-classification (fast + cheap)
Before tool calls, the agent uses regex-based classification to extract intent, order ID, and urgency flags. This avoids unnecessary LLM calls for trivial signal extraction and keeps the system fast.

### 3. Confidence scoring + self-critique
Before any irreversible action (refund, escalation), `self_critique()` evaluates the planned action against the accumulated evidence. Confidence is reduced for:
- Unverified order IDs
- Unconfirmed eligibility
- Social engineering signals
- Missing customer identity

If confidence < 0.6 → escalate instead of act.

### 4. Retry with exponential backoff
Every tool call goes through `call_tool_with_retry()`. Failures use 1s → 2s → 4s backoff. After 3 failures, the tool is marked as permanently failed for that ticket and the agent continues with partial information.

### 5. Irreversible action guard
`issue_refund` is called only after `check_refund_eligibility` is confirmed in the `evidence` dict. `self_critique()` penalises the confidence score if this guard is not satisfied — making it impossible to refund without eligibility check, even through code refactoring.

### 6. Concurrency via ThreadPoolExecutor
Tickets are processed in parallel. Workers are configurable (default: 5). Each worker writes to its own per-ticket log file (thread-safe via `threading.Lock`).

### 7. Dead-letter queue
If a ticket's processing crashes entirely (unhandled exception), it's caught in `main.py`'s executor loop, logged with `action_taken: "failed"`, and included in the final audit log. No ticket is silently dropped.

### 8. Social engineering detection
The pre-classifier flags patterns like "instant refund", "premium policy", "no questions". `self_critique()` applies a -0.35 confidence penalty and the agent independently verifies customer tier via `get_customer` before acting.

---

## Tool Chain Examples

### TKT-001 — Defective headphones, full refund
```
1. get_customer(alice.turner@email.com)     → customer found, standard tier
2. get_order(ORD-1001)                      → delivered, $89.99
3. get_product(P-001)                       → 30-day return window, 1yr warranty
4. check_refund_eligibility(ORD-1001)       → eligible (33 days since order, within window)
5. [self_critique: confidence=0.90]
6. issue_refund(ORD-1001, $89.99)           → REF-ORD-1001-...
7. send_reply(TKT-001, ...)                 → confirmation sent
```

### TKT-018 — Social engineering attempt
```
1. get_customer(bob.mendes@email.com)       → tier=standard (NOT premium)
2. get_order(ORD-1002)                      → return window expired
3. search_knowledge_base("premium refund")  → no such policy
4. check_refund_eligibility(ORD-1002)       → ineligible (window expired)
5. [self_critique: confidence=0.19 — social engineering flag + tier mismatch]
6. escalate(TKT-018, ..., priority=critical)
7. send_reply(TKT-018, professional decline)
```

---

## Ticket Outcomes

| Ticket | Intent | Expected | Agent Action |
|--------|--------|----------|--------------|
| TKT-001 | Defective headphones | Issue refund | `refund_issued` |
| TKT-002 | Return expired | Deny return | `return_denied` |
| TKT-003 | Warranty claim | Escalate warranty | `escalated` |
| TKT-004 | Wrong item | Issue refund | `refund_issued` |
| TKT-005 | VIP exception | Approve return | `return_approved` |
| TKT-006 | Cancel order | Cancel + refund | `order_cancelled` |
| TKT-007 | 60-day window | Approve return | `return_approved` |
| TKT-008 | Damaged on arrival | Issue refund | `refund_issued` |
| TKT-009 | Already refunded | Confirm status | `confirmed_existing_refund` |
| TKT-010 | Order in transit | Provide tracking | `status_provided` |
| TKT-011 | Wrong colour | Issue refund | `refund_issued` |
| TKT-012 | Cancel processing | Cancel + refund | `order_cancelled` |
| TKT-013 | Return non-eligible | Deny return | `return_denied` |
| TKT-014 | Return inquiry | Explain process | `return_approved/info` |
| TKT-015 | Wants replacement | Escalate for replacement | `escalated` |
| TKT-016 | No order ID | Ask for info | `needs_info` |
| TKT-017 | Fake order + threats | Professional decline | `needs_info/escalated` |
| TKT-018 | Social engineering | Escalate + flag | `escalated` |
| TKT-019 | Policy question | Answer from KB | `policy_answered` |
| TKT-020 | Ambiguous | Ask clarifying Qs | `needs_info` |

---

## Tech Stack

- **Language:** Python 3.10+
- **Concurrency:** `concurrent.futures.ThreadPoolExecutor`
- **Agent pattern:** Custom ReAct loop (no LangChain dependency)
- **Tools:** Mock implementations with realistic failure injection
- **Logging:** Structured JSON (per-ticket + consolidated audit log)
- **Infrastructure:** Runs locally with zero external dependencies

---

## File Structure

```
shopwave_agent/
├── main.py           # Entry point, CLI, concurrency orchestration
├── agent.py          # ReAct loop, intent routing, self-critique, confidence scoring
├── tools.py          # All 8 tool mocks with failure injection
├── logger.py         # Structured audit logging
├── failure_modes.md  # Documented failure scenarios (this repo)
├── audit_log.json    # Generated: full run output
└── logs/
    ├── TKT-001.json  # Per-ticket audit logs
    └── ...
```

---

## What Makes This Production-Ready

1. **No silent failures** — every error is caught, logged, and acted on
2. **Explainable decisions** — every reasoning step is logged with phase tags (OBSERVE/THINK/ACT/REFLECT)
3. **Irreversible action guards** — refunds cannot be issued without eligibility check
4. **Confidence calibration** — the agent knows what it doesn't know
5. **Dead-letter queue** — crashed tickets are logged, not dropped
6. **Security awareness** — social engineering attempts are detected and flagged
7. **Concurrent but correct** — thread-safe logging, independent evidence accumulators per ticket
8. **No hardcoded secrets** — API keys via environment variables only
