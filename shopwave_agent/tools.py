"""
tools.py — Mock tool implementations for ShopWave support agent.

Each tool simulates realistic failures: random timeouts, malformed data,
partial responses. This is intentional — the agent must handle them gracefully.

Design decision: Tools are stateless functions. The agent owns the state.
Failure injection is controlled via TOOL_FAILURE_RATE env var (default: 0.15).
"""

import json
import random
import time
import os
from datetime import datetime, timezone
from typing import Any

# ── Failure injection config ────────────────────────────────────────────────
FAILURE_RATE = float(os.getenv("TOOL_FAILURE_RATE", "0.15"))

# ── Static mock data ─────────────────────────────────────────────────────────

CUSTOMERS = {
    "alice.turner@email.com":   {"customer_id": "C-001", "name": "Alice Turner",  "tier": "standard", "total_orders": 8,  "vip_exception": False},
    "bob.mendes@email.com":     {"customer_id": "C-002", "name": "Bob Mendes",    "tier": "standard", "total_orders": 3,  "vip_exception": False},
    "carol.nguyen@email.com":   {"customer_id": "C-003", "name": "Carol Nguyen",  "tier": "standard", "total_orders": 5,  "vip_exception": False},
    "david.park@email.com":     {"customer_id": "C-004", "name": "David Park",    "tier": "standard", "total_orders": 2,  "vip_exception": False},
    "emma.collins@email.com":   {"customer_id": "C-005", "name": "Emma Collins",  "tier": "vip",      "total_orders": 22, "vip_exception": True},
    "frank.osei@email.com":     {"customer_id": "C-006", "name": "Frank Osei",    "tier": "standard", "total_orders": 1,  "vip_exception": False},
    "grace.patel@email.com":    {"customer_id": "C-007", "name": "Grace Patel",   "tier": "premium",  "total_orders": 14, "vip_exception": False},
    "henry.marsh@email.com":    {"customer_id": "C-008", "name": "Henry Marsh",   "tier": "standard", "total_orders": 4,  "vip_exception": False},
    "irene.castillo@email.com": {"customer_id": "C-009", "name": "Irene Castillo","tier": "standard", "total_orders": 6,  "vip_exception": False},
    "james.wu@email.com":       {"customer_id": "C-010", "name": "James Wu",      "tier": "standard", "total_orders": 3,  "vip_exception": False},
}

ORDERS = {
    "ORD-1001": {"order_id": "ORD-1001", "customer_email": "alice.turner@email.com",   "product_id": "P-001", "product_name": "ProSound Headphones",     "amount": 89.99,  "status": "delivered",   "ordered_at": "2024-02-10T10:00:00Z", "delivered_at": "2024-02-14T12:00:00Z", "tracking": "TRK-10010"},
    "ORD-1002": {"order_id": "ORD-1002", "customer_email": "bob.mendes@email.com",     "product_id": "P-002", "product_name": "PulseX Smart Watch",       "amount": 199.99, "status": "delivered",   "ordered_at": "2024-03-01T09:00:00Z", "delivered_at": "2024-03-04T14:00:00Z", "tracking": "TRK-10020"},
    "ORD-1003": {"order_id": "ORD-1003", "customer_email": "carol.nguyen@email.com",   "product_id": "P-003", "product_name": "BrewMaster Coffee Maker",  "amount": 129.99, "status": "delivered",   "ordered_at": "2024-02-20T11:00:00Z", "delivered_at": "2024-02-25T10:00:00Z", "tracking": "TRK-10030"},
    "ORD-1004": {"order_id": "ORD-1004", "customer_email": "david.park@email.com",     "product_id": "P-004", "product_name": "AeroRun Running Shoes",    "amount": 74.99,  "status": "delivered",   "ordered_at": "2024-03-10T08:00:00Z", "delivered_at": "2024-03-13T16:00:00Z", "tracking": "TRK-10040"},
    "ORD-1005": {"order_id": "ORD-1005", "customer_email": "emma.collins@email.com",   "product_id": "P-005", "product_name": "BassBoost BT Speaker x2",  "amount": 159.98, "status": "delivered",   "ordered_at": "2023-12-01T10:00:00Z", "delivered_at": "2023-12-05T10:00:00Z", "tracking": "TRK-10050"},
    "ORD-1006": {"order_id": "ORD-1006", "customer_email": "frank.osei@email.com",     "product_id": "P-006", "product_name": "ErgoDesk Lamp",            "amount": 39.99,  "status": "processing",  "ordered_at": "2024-03-14T15:00:00Z", "delivered_at": None,                   "tracking": None},
    "ORD-1007": {"order_id": "ORD-1007", "customer_email": "grace.patel@email.com",    "product_id": "P-007", "product_name": "NexStand Laptop Stand",    "amount": 54.99,  "status": "delivered",   "ordered_at": "2024-01-14T10:00:00Z", "delivered_at": "2024-01-18T12:00:00Z", "tracking": "TRK-10070"},
    "ORD-1008": {"order_id": "ORD-1008", "customer_email": "henry.marsh@email.com",    "product_id": "P-006", "product_name": "ErgoDesk Lamp",            "amount": 39.99,  "status": "delivered",   "ordered_at": "2024-03-05T10:00:00Z", "delivered_at": "2024-03-09T09:00:00Z", "tracking": "TRK-10080"},
    "ORD-1009": {"order_id": "ORD-1009", "customer_email": "irene.castillo@email.com", "product_id": "P-001", "product_name": "ProSound Headphones",     "amount": 89.99,  "status": "refunded",    "ordered_at": "2024-03-01T10:00:00Z", "delivered_at": "2024-03-05T10:00:00Z", "tracking": "TRK-10090"},
    "ORD-1010": {"order_id": "ORD-1010", "customer_email": "james.wu@email.com",       "product_id": "P-003", "product_name": "BrewMaster Coffee Maker",  "amount": 129.99, "status": "shipped",     "ordered_at": "2024-03-12T10:00:00Z", "delivered_at": None,                   "tracking": "TRK-88291"},
    "ORD-1011": {"order_id": "ORD-1011", "customer_email": "alice.turner@email.com",   "product_id": "P-002", "product_name": "PulseX Smart Watch (Black)","amount": 199.99, "status": "delivered",  "ordered_at": "2024-03-08T10:00:00Z", "delivered_at": "2024-03-12T10:00:00Z", "tracking": "TRK-10110"},
    "ORD-1012": {"order_id": "ORD-1012", "customer_email": "carol.nguyen@email.com",   "product_id": "P-004", "product_name": "AeroRun Running Shoes",    "amount": 74.99,  "status": "processing",  "ordered_at": "2024-03-14T10:00:00Z", "delivered_at": None,                   "tracking": None},
    "ORD-1013": {"order_id": "ORD-1013", "customer_email": "grace.patel@email.com",    "product_id": "P-005", "product_name": "BassBoost BT Speaker",     "amount": 79.99,  "status": "delivered",   "ordered_at": "2024-01-10T10:00:00Z", "delivered_at": "2024-01-14T10:00:00Z", "tracking": "TRK-10130"},
    "ORD-1014": {"order_id": "ORD-1014", "customer_email": "henry.marsh@email.com",    "product_id": "P-002", "product_name": "PulseX Smart Watch",       "amount": 199.99, "status": "delivered",   "ordered_at": "2024-03-10T10:00:00Z", "delivered_at": "2024-03-14T10:00:00Z", "tracking": "TRK-10140"},
    "ORD-1015": {"order_id": "ORD-1015", "customer_email": "emma.collins@email.com",   "product_id": "P-003", "product_name": "BrewMaster Coffee Maker",  "amount": 129.99, "status": "delivered",   "ordered_at": "2024-03-10T10:00:00Z", "delivered_at": "2024-03-14T10:00:00Z", "tracking": "TRK-10150"},
}

PRODUCTS = {
    "P-001": {"product_id": "P-001", "name": "ProSound Headphones",     "category": "electronics", "warranty_days": 365, "return_window_days": 30, "returnable_if_registered": False},
    "P-002": {"product_id": "P-002", "name": "PulseX Smart Watch",      "category": "electronics", "warranty_days": 365, "return_window_days": 15, "returnable_if_registered": False},
    "P-003": {"product_id": "P-003", "name": "BrewMaster Coffee Maker", "category": "appliances",  "warranty_days": 730, "return_window_days": 30, "returnable_if_registered": False},
    "P-004": {"product_id": "P-004", "name": "AeroRun Running Shoes",   "category": "footwear",    "warranty_days": 180, "return_window_days": 30, "returnable_if_registered": True},
    "P-005": {"product_id": "P-005", "name": "BassBoost BT Speaker",    "category": "electronics", "warranty_days": 365, "return_window_days": 30, "returnable_if_registered": False},
    "P-006": {"product_id": "P-006", "name": "ErgoDesk Lamp",           "category": "home",        "warranty_days": 365, "return_window_days": 30, "returnable_if_registered": True},
    "P-007": {"product_id": "P-007", "name": "NexStand Laptop Stand",   "category": "accessories", "warranty_days": 365, "return_window_days": 60, "returnable_if_registered": True},
}

KNOWLEDGE_BASE = {
    "return policy": "Standard return window is 30 days from delivery. Some products (laptop accessories) have 60-day windows. Items must be unused and in original packaging unless defective.",
    "refund policy": "Refunds are issued to the original payment method within 5-7 business days after the return is received and inspected.",
    "warranty": "Electronics carry a 1-year manufacturer warranty. Appliances carry a 2-year warranty. Warranty covers manufacturing defects, not accidental damage.",
    "exchange process": "Exchanges can be requested within the return window. Subject to stock availability. We ship the replacement first for VIP customers.",
    "damaged on arrival": "If an item arrives damaged, we issue a full refund without requiring the item to be returned. Photos are helpful but not mandatory.",
    "cancellation": "Orders in 'processing' status can be cancelled. Orders that have shipped cannot be cancelled — initiate a return instead.",
    "vip policy": "VIP customers have pre-approved return exceptions and receive priority handling. Check customer notes for VIP exception flags.",
    "social engineering": "ShopWave does not have an 'instant refund for premium members' policy. All refunds follow standard eligibility checks regardless of claimed tier.",
}

# ── Failure simulation helper ─────────────────────────────────────────────────

class ToolTimeoutError(Exception):
    pass

class ToolDataError(Exception):
    pass

def _maybe_fail(tool_name: str) -> None:
    """Randomly inject realistic failures based on configured rate."""
    roll = random.random()
    if roll < FAILURE_RATE * 0.5:
        time.sleep(0.1)  # simulate timeout
        raise ToolTimeoutError(f"{tool_name} timed out after 30s")
    elif roll < FAILURE_RATE:
        raise ToolDataError(f"{tool_name} returned malformed response (upstream API error)")

# ── READ tools ────────────────────────────────────────────────────────────────

def get_order(order_id: str) -> dict[str, Any]:
    """Fetch order details by order ID."""
    _maybe_fail("get_order")
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "not_found", "message": f"Order {order_id} does not exist"}
    return {"success": True, "order": dict(order)}


def get_customer(email: str) -> dict[str, Any]:
    """Fetch customer profile by email address."""
    _maybe_fail("get_customer")
    customer = CUSTOMERS.get(email.lower())
    if not customer:
        return {"error": "not_found", "message": f"No customer found for email {email}"}
    return {"success": True, "customer": dict(customer)}


def get_product(product_id: str) -> dict[str, Any]:
    """Fetch product metadata including warranty and return window."""
    _maybe_fail("get_product")
    product = PRODUCTS.get(product_id)
    if not product:
        return {"error": "not_found", "message": f"Product {product_id} not found"}
    return {"success": True, "product": dict(product)}


def search_knowledge_base(query: str) -> dict[str, Any]:
    """Semantic-style search over policy and FAQ content."""
    _maybe_fail("search_knowledge_base")
    query_lower = query.lower()
    results = []
    for key, value in KNOWLEDGE_BASE.items():
        if any(term in query_lower for term in key.split()):
            results.append({"topic": key, "content": value})
    if not results:
        # Return best-guess fallback
        results = [{"topic": "general", "content": "Please contact support for specific policy questions not covered in the FAQ."}]
    return {"success": True, "results": results}


def check_refund_eligibility(order_id: str, reference_date: str = None) -> dict[str, Any]:
    """
    Check whether an order qualifies for a refund.
    This tool may throw errors — agent must handle gracefully.
    """
    _maybe_fail("check_refund_eligibility")
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "not_found", "message": f"Order {order_id} not found"}

    product = PRODUCTS.get(order["product_id"], {})
    return_window = product.get("return_window_days", 30)

    # Calculate days since delivery
    if not order.get("delivered_at"):
        return {"eligible": False, "reason": "Order not yet delivered — cannot process refund"}

    delivered_dt = datetime.fromisoformat(order["delivered_at"].replace("Z", "+00:00"))
    # Use reference_date if provided (agent passes ticket creation date for accurate window calculation)
    # This ensures eligibility is evaluated as of the ticket submission date, not execution date
    if reference_date:
        now = datetime.fromisoformat(reference_date.replace("Z", "+00:00"))
    else:
        now = datetime.now(timezone.utc)
    days_since = (now - delivered_dt).days

    if order["status"] == "refunded":
        return {"eligible": False, "reason": "Refund already processed for this order"}

    if order["status"] == "processing":
        return {"eligible": True, "reason": "Order in processing — cancellation and full refund available", "type": "cancellation"}

    if days_since <= return_window:
        return {"eligible": True, "reason": f"Within {return_window}-day return window ({days_since} days since delivery)", "amount": order["amount"], "type": "standard_return"}

    return {"eligible": False, "reason": f"Return window expired ({days_since} days since delivery, window is {return_window} days)"}


# ── WRITE tools ───────────────────────────────────────────────────────────────

def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
    """
    IRREVERSIBLE — issues a refund. Agent must verify eligibility first.
    In production this would call a payments API.
    """
    _maybe_fail("issue_refund")
    order = ORDERS.get(order_id)
    if not order:
        return {"error": "not_found", "message": f"Order {order_id} not found"}
    # Simulate success
    return {
        "success": True,
        "refund_id": f"REF-{order_id}-{int(time.time())}",
        "amount": amount,
        "order_id": order_id,
        "message": f"Refund of ${amount:.2f} initiated. Customer will receive funds in 5–7 business days.",
    }


def send_reply(ticket_id: str, message: str) -> dict[str, Any]:
    """Send a reply message to the customer."""
    _maybe_fail("send_reply")
    return {
        "success": True,
        "ticket_id": ticket_id,
        "message_sent": message,
        "sent_at": datetime.utcnow().isoformat() + "Z",
    }


def escalate(ticket_id: str, summary: str, priority: str) -> dict[str, Any]:
    """Route ticket to a human agent with full structured context."""
    _maybe_fail("escalate")
    valid_priorities = {"low", "medium", "high", "critical"}
    if priority not in valid_priorities:
        priority = "medium"
    return {
        "success": True,
        "ticket_id": ticket_id,
        "escalated_to": "human_queue",
        "priority": priority,
        "summary": summary,
        "escalated_at": datetime.utcnow().isoformat() + "Z",
    }


# ── Tool registry ─────────────────────────────────────────────────────────────
# Used by the agent to dispatch tool calls by name.

TOOL_REGISTRY: dict[str, callable] = {
    "get_order":               get_order,
    "get_customer":            get_customer,
    "get_product":             get_product,
    "search_knowledge_base":   search_knowledge_base,
    "check_refund_eligibility": check_refund_eligibility,
    "issue_refund":            issue_refund,
    "send_reply":              send_reply,
    "escalate":                escalate,
}
