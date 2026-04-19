# Failure Mode Analysis — ShopWave Support Agent

## Overview

The agent is designed to **never crash silently**. Every failure is caught, logged, and either retried, recovered from, or escalated with full context. Below are the documented failure modes and how the system handles each.

---

## Failure Mode 1: Tool Timeout / Network Error

**What happens:** A tool call (e.g. `get_order`, `check_refund_eligibility`) takes too long or throws a `ToolTimeoutError`.

**Root cause:** Upstream API unresponsive, database lock, or network partition.

**System response:**
1. The error is caught in `call_tool_with_retry()` in `agent.py`
2. The failure is immediately logged with timestamp and attempt number
3. The agent waits with **exponential backoff**: 1s → 2s → 4s
4. After `MAX_RETRIES` (default: 3) attempts, the tool is marked as permanently failed
5. The agent continues reasoning with **partial information** rather than crashing
6. If the failed tool was critical (e.g. eligibility check before a refund), the ticket is **escalated** with a clear human-readable summary

**Example log entry:**
```json
{
  "tool": "check_refund_eligibility",
  "inputs": {"order_id": "ORD-1001"},
  "attempt": 2,
  "status": "error",
  "error": "check_refund_eligibility timed out after 30s",
  "duration_ms": 102.3
}
```

**Why this design:** Silent failure = invisible bugs in production. Every retry attempt is logged so engineers can identify flaky tools from audit logs.

---

## Failure Mode 2: Malformed / Unexpected Tool Response

**What happens:** A tool returns structurally invalid data — missing fields, wrong types, or an error response instead of the expected schema.

**Root cause:** API contract broken by upstream service update, or partial response from a degraded service.

**System response:**
1. Each tool result is defensively accessed with `.get()` — no KeyError crashes
2. Missing fields fall back to safe defaults (e.g. `order.get("amount", 0)`)
3. If a critical field (e.g. refund `amount`) is missing, confidence scoring penalises the decision
4. A `ToolDataError` is thrown and handled identically to timeouts (retry → log → partial continue)
5. The `evidence` dict accumulates only validated data — acting on bad data is prevented

**Example:** `check_refund_eligibility` returns `{"error": "malformed"}` instead of eligibility status:
- Agent logs the error
- Treats it as "eligibility unknown"
- Confidence drops below threshold
- Ticket escalated rather than auto-refunded

**Why this design:** The irreversible nature of `issue_refund` means we must never act on bad data. Confidence calibration is the guard.

---

## Failure Mode 3: Order Not Found / Unknown Customer (Invalid Data)

**What happens:** Customer provides a non-existent order ID (e.g. `ORD-9999` in TKT-017) or uses an unregistered email.

**Root cause:** Typo, fraud attempt, or test account.

**System response:**
1. `get_order` returns `{"error": "not_found"}` — no exception thrown
2. Agent sets `evidence["order_not_found"] = True`
3. `self_critique()` detects this flag and reduces confidence by 0.40
4. For financial actions (refund, return), confidence will be below threshold
5. Agent sends a professional, non-accusatory reply asking for correct order details
6. If threatening language is also present (TKT-017), the response remains calm but the ticket is flagged

**Customer response sent:**
> "I wasn't able to locate an order with that ID in our system. Could you double-check the order number in your confirmation email? Once I have the correct details, I'll get this sorted right away."

**Why this design:** We don't accuse customers of fraud. But we also don't issue refunds for orders that don't exist. The confidence system prevents the latter without requiring explicit IF/ELSE fraud detection.

---

## Failure Mode 4: Low Confidence / Ambiguous Intent

**What happens:** The agent cannot reliably determine what the customer wants (e.g. TKT-020: "my thing is broken pls help").

**System response:**
1. Pre-classifier sets `intent = "unknown"`
2. `self_critique()` would assign low confidence to any financial action
3. Agent routes to `_handle_ambiguous()` which asks targeted clarifying questions
4. Knowledge base is still searched to prepare for likely follow-up
5. The clarification request is logged as `needs_info` — not dropped

**Why this design:** Asking for clarification IS a valid resolution. The alternative — guessing wrong and issuing a refund for the wrong order — is catastrophically worse.

---

## Failure Mode 5: Social Engineering / Policy Manipulation (TKT-018)

**What happens:** Customer falsely claims a non-existent policy ("premium members get instant refunds without questions").

**System response:**
1. Pre-classifier detects `potential_social_engineering` flag from keywords
2. `self_critique()` applies a -0.35 confidence penalty for this flag
3. Agent independently verifies customer tier via `get_customer` tool — finds `tier = "standard"`
4. Knowledge base is searched to confirm no such policy exists
5. Even if all other checks passed, confidence will be below threshold
6. Ticket is escalated with a detailed summary flagging the manipulation attempt
7. Customer receives a professional reply with no accusation, just policy explanation

**Why this design:** The agent never accuses customers directly. But it also never acts on claimed-but-unverified policies. Independent verification via tool calls is the protection mechanism.

---

## Failure Mode 6: Irreversible Action Guard (issue_refund safety)

**What happens:** Some code path attempts to call `issue_refund` without first checking eligibility.

**System response:**
1. `self_critique()` specifically checks `evidence["eligibility_checked"]`
2. If eligibility was not confirmed, confidence is reduced by 0.30
3. This typically drops confidence below the 0.60 threshold
4. The agent escalates rather than proceeding with the irreversible action
5. This is enforced at the reasoning level — not just code-level guards

**Why this design:** Production systems that handle money must have belt-AND-suspenders safety. Code guards can be bypassed by refactoring. Reasoning-level guards cannot.

---

## Summary Table

| Failure Mode | Detection | Recovery Strategy | Logged? |
|---|---|---|---|
| Tool timeout | `ToolTimeoutError` | Retry with backoff (3x) | ✅ Every attempt |
| Malformed data | `ToolDataError` | Retry → partial info | ✅ Every attempt |
| Order not found | `error.not_found` response | Ask for correct details | ✅ |
| Customer not found | `error.not_found` response | Request identity verification | ✅ |
| Low confidence | `self_critique()` score < 0.6 | Escalate instead of act | ✅ With critique |
| Social engineering | Keyword flag + tier mismatch | Escalate + flag summary | ✅ |
| Ambiguous ticket | `intent = "unknown"` | Clarifying questions | ✅ |
| Irreversible action without eligibility | Missing evidence flag | Confidence penalty → escalate | ✅ |
| Complete agent crash | `ThreadPoolExecutor` catch-all | Dead-letter log entry | ✅ |
