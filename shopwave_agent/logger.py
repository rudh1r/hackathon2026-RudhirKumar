"""
logger.py — Structured audit logging for the ShopWave support agent.

Every decision the agent makes is logged with:
  - The reasoning step (Observe / Think / Act / Reflect)
  - Tool calls and their results (including failures and retries)
  - Final decision and confidence score
  - Outcome (resolved / escalated / failed)

Design decision: We write a running log per ticket, then consolidate into
a single audit_log.json at the end. This means we never lose partial work
if the process crashes mid-run.
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

LOG_DIR = os.getenv("LOG_DIR", "logs")
_lock = threading.Lock()  # thread-safe writes for concurrent ticket processing


class TicketLogger:
    """
    Captures every reasoning step and tool interaction for one ticket.
    Call .finalize() to write the log to disk.
    """

    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.started_at = _now()
        self.steps: list[dict] = []
        self.tool_calls: list[dict] = []
        self.final_outcome: dict = {}

    # ── Logging methods ───────────────────────────────────────────────────────

    def observe(self, observation: str) -> None:
        """Log what the agent perceives about the ticket."""
        self._add_step("OBSERVE", observation)

    def think(self, reasoning: str) -> None:
        """Log the agent's reasoning before taking an action."""
        self._add_step("THINK", reasoning)

    def act(self, action: str) -> None:
        """Log the action the agent decides to take."""
        self._add_step("ACT", action)

    def reflect(self, reflection: str) -> None:
        """Log the agent's self-critique after acting."""
        self._add_step("REFLECT", reflection)

    def tool_call(
        self,
        tool_name: str,
        inputs: dict,
        result: Any,
        attempt: int = 1,
        error: str | None = None,
        duration_ms: float = 0,
    ) -> None:
        """Log a single tool invocation with its outcome."""
        entry = {
            "tool": tool_name,
            "inputs": inputs,
            "attempt": attempt,
            "duration_ms": round(duration_ms, 1),
            "timestamp": _now(),
        }
        if error:
            entry["status"] = "error"
            entry["error"] = error
        else:
            entry["status"] = "success"
            entry["result"] = result
        self.tool_calls.append(entry)

    def set_outcome(
        self,
        action_taken: str,
        resolution: str,
        confidence: float,
        customer_message: str = "",
    ) -> None:
        """Record the final resolution decision."""
        self.final_outcome = {
            "action_taken": action_taken,
            "resolution": resolution,
            "confidence": round(confidence, 2),
            "customer_message_sent": customer_message,
            "completed_at": _now(),
        }

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "started_at": self.started_at,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "final_outcome": self.final_outcome,
            "tool_call_count": len(self.tool_calls),
            "successful_tool_calls": sum(1 for t in self.tool_calls if t["status"] == "success"),
            "failed_tool_calls": sum(1 for t in self.tool_calls if t["status"] == "error"),
        }

    def finalize(self) -> None:
        """Write this ticket's log to disk (thread-safe)."""
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"{self.ticket_id}.json")
        with _lock:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add_step(self, phase: str, content: str) -> None:
        self.steps.append({"phase": phase, "content": content, "timestamp": _now()})


def consolidate_logs(output_path: str = "audit_log.json") -> None:
    """
    Merge all per-ticket logs into a single audit_log.json.
    Called after all tickets have been processed.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logs = []
    for fname in sorted(os.listdir(LOG_DIR)):
        if fname.endswith(".json") and fname.startswith("TKT-"):
            with open(os.path.join(LOG_DIR, fname)) as f:
                logs.append(json.load(f))

    summary = {
        "generated_at": _now(),
        "total_tickets": len(logs),
        "resolved_autonomously": sum(1 for l in logs if l.get("final_outcome", {}).get("action_taken") not in ("escalated", "failed", "needs_info")),
        "escalated": sum(1 for l in logs if l.get("final_outcome", {}).get("action_taken") == "escalated"),
        "needs_info": sum(1 for l in logs if l.get("final_outcome", {}).get("action_taken") == "needs_info"),
        "failed": sum(1 for l in logs if l.get("final_outcome", {}).get("action_taken") == "failed"),
        "avg_tool_calls": round(sum(l.get("tool_call_count", 0) for l in logs) / max(len(logs), 1), 1),
        "tickets": logs,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"AUDIT LOG: {output_path}")
    print(f"  Total tickets   : {summary['total_tickets']}")
    print(f"  Auto-resolved   : {summary['resolved_autonomously']}")
    print(f"  Escalated       : {summary['escalated']}")
    print(f"  Needs more info : {summary['needs_info']}")
    print(f"  Failed          : {summary['failed']}")
    print(f"  Avg tool calls  : {summary['avg_tool_calls']}")
    print(f"{'='*60}\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
