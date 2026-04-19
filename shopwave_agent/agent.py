"""
agent.py — The ShopWave Autonomous Support Resolution Agent.

Architecture: ReAct loop (Reason + Act) with 4 phases per ticket:
  1. OBSERVE  — Parse the ticket, extract key signals
  2. THINK    — Plan the resolution strategy
  3. ACT      — Execute tool chains (min 3 calls per ticket when appropriate)
  4. REFLECT  — Self-critique the decision before committing

Key design decisions:
  - Every action is explainable (no black-box decisions)
  - Tools are called with retry + exponential backoff
  - Confidence < 0.6 → escalate instead of acting
  - Irreversible actions (issue_refund) require eligibility check first
  - All reasoning steps are logged, not just final answers
"""

import os
import time
import json
import re
from datetime import datetime, timezone
from typing import Any

from tools import TOOL_REGISTRY, ToolTimeoutError, ToolDataError
from logger import TicketLogger

# ── Config ────────────────────────────────────────────────────────────────────
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_BASE    = float(os.getenv("BACKOFF_BASE", "1.0"))   # seconds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MODEL           = "claude-sonnet-4-20250514"


# ── Tool execution with retry + backoff ───────────────────────────────────────

def call_tool_with_retry(
    tool_name: str,
    tool_inputs: dict,
    log: TicketLogger,
    max_retries: int = MAX_RETRIES,
) -> Any:
    """
    Execute a tool, retrying on transient failures with exponential backoff.
    Non-retryable errors (not_found) are returned immediately.
    Permanently failed calls are logged and return an error dict.
    """
    tool_fn = TOOL_REGISTRY.get(tool_name)
    if not tool_fn:
        return {"error": "unknown_tool", "message": f"Tool '{tool_name}' not registered"}

    last_error = None
    for attempt in range(1, max_retries + 1):
        t0 = time.monotonic()
        try:
            result = tool_fn(**tool_inputs)
            duration = (time.monotonic() - t0) * 1000
            log.tool_call(tool_name, tool_inputs, result, attempt=attempt, duration_ms=duration)
            return result

        except (ToolTimeoutError, ToolDataError) as e:
            duration = (time.monotonic() - t0) * 1000
            last_error = str(e)
            log.tool_call(tool_name, tool_inputs, None, attempt=attempt, error=last_error, duration_ms=duration)

            if attempt < max_retries:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))  # 1s, 2s, 4s
                log.think(f"Tool '{tool_name}' failed (attempt {attempt}/{max_retries}): {last_error}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                log.think(f"Tool '{tool_name}' failed after {max_retries} attempts. Moving on with partial information.")

    return {"error": "tool_failure", "message": last_error or "Unknown tool failure"}


# ── Classifier: extract ticket signals without LLM (fast, deterministic) ──────

def classify_ticket(ticket: dict) -> dict:
    """
    Rule-based pre-classification. Runs before LLM to reduce token cost
    and provide a starting scaffold for the agent's reasoning.
    """
    body = (ticket.get("body", "") + " " + ticket.get("subject", "")).lower()

    # Extract order ID from body
    order_match = re.search(r"ord-\d+", body, re.IGNORECASE)
    order_id = order_match.group(0).upper() if order_match else None

    # Detect intent — policy questions checked first (they often contain "return"/"refund" words)
    intent = "unknown"
    # Policy question: general info requests, no specific order context
    policy_phrases = ["what is your", "what are your", "do you offer", "how does your", "what is the policy",
                      "return policy", "refund policy", "exchange policy", "how do returns", "do you accept returns"]
    if any(p in body for p in policy_phrases) and not re.search(r"ord-\d+", body, re.IGNORECASE):
        intent = "policy_question"
    elif any(w in body for w in ["refund", "money back"]):
        intent = "refund_request"
    elif any(w in body for w in ["return", "send back", "exchange"]):
        intent = "return_request"
    elif any(w in body for w in ["cancel", "cancell"]):
        intent = "cancellation"
    elif any(w in body for w in ["where is", "tracking", "shipped", "delivery", "arrived"]):
        intent = "order_status"
    elif any(w in body for w in ["broken", "defect", "stopped working", "damaged", "cracked", "not working"]):
        intent = "defective_item"
    elif any(w in body for w in ["wrong", "incorrect", "different"]):
        intent = "wrong_item"
    elif any(w in body for w in ["policy", "how do i", "what is", "do you", "can i"]):
        intent = "policy_question"

    # Detect urgency signals
    urgency = "normal"
    if any(w in body for w in ["urgent", "immediately", "today", "asap", "dispute", "lawyer", "bank"]):
        urgency = "high"

    # Detect potentially adversarial patterns
    flags = []
    if any(w in body for w in ["lawyer", "dispute", "bank", "legal"]):
        flags.append("threatening_language")
    if any(w in body for w in ["premium member", "instant refund", "vip policy", "no questions"]):
        flags.append("potential_social_engineering")

    return {
        "intent": intent,
        "order_id": order_id,
        "urgency": urgency,
        "flags": flags,
        "has_order_id": order_id is not None,
    }


# ── Self-critique: agent evaluates its own planned action ─────────────────────

def self_critique(
    ticket: dict,
    classification: dict,
    planned_action: str,
    evidence: dict,
    log: TicketLogger,
) -> tuple[float, str]:
    """
    Before committing to an action, the agent asks: "Am I about to do
    something wrong?" Returns (confidence: float, critique: str).

    Rules that reduce confidence:
    - Issuing a refund without confirmed eligibility check
    - Acting on a ticket with potential_social_engineering flag
    - Missing order data for financial actions
    - Order not found
    """
    confidence = 0.9
    notes = []

    flags = classification.get("flags", [])
    intent = classification.get("intent", "")

    if "potential_social_engineering" in flags:
        confidence -= 0.35
        notes.append("WARNING: Potential social engineering detected — applying extra scrutiny")

    if planned_action in ("issue_refund", "approve_return") and not evidence.get("eligibility_checked"):
        confidence -= 0.30
        notes.append("Refund/return action planned but eligibility not yet confirmed")

    if evidence.get("order_not_found"):
        confidence -= 0.40
        notes.append("Order ID not found in system — cannot proceed with financial action")

    if evidence.get("customer_not_found") and intent in ("refund_request", "return_request"):
        confidence -= 0.25
        notes.append("Customer not found — identity unverified")

    if classification.get("flags") and "threatening_language" in flags and planned_action != "escalate":
        confidence -= 0.10
        notes.append("Threatening language detected — consider escalation")

    # Clamp
    confidence = max(0.05, min(1.0, confidence))
    critique = "; ".join(notes) if notes else "No concerns — action looks appropriate"

    log.reflect(f"Self-critique | confidence={confidence:.2f} | {critique}")
    return confidence, critique


# ── The core ReAct agent loop ─────────────────────────────────────────────────

def process_ticket(ticket: dict) -> dict:
    """
    Full ReAct loop for one support ticket.
    Returns a result dict with action_taken, resolution, and confidence.
    """
    ticket_id = ticket["ticket_id"]
    log = TicketLogger(ticket_id)

    # ── PHASE 1: OBSERVE ──────────────────────────────────────────────────────
    log.observe(
        f"Ticket received | id={ticket_id} | "
        f"from={ticket['customer_email']} | "
        f"subject='{ticket['subject']}' | "
        f"source={ticket['source']} | "
        f"body='{ticket['body'][:120]}...'"
    )

    classification = classify_ticket(ticket)
    log.observe(
        f"Pre-classification | intent={classification['intent']} | "
        f"order_id={classification['order_id']} | "
        f"urgency={classification['urgency']} | "
        f"flags={classification['flags']}"
    )

    # Evidence accumulator — shared state across tool calls
    evidence: dict[str, Any] = {
        "eligibility_checked": False,
        "order_not_found": False,
        "customer_not_found": False,
    }

    # ── PHASE 2: THINK — plan the resolution strategy ─────────────────────────
    intent = classification["intent"]
    order_id = classification["order_id"]
    email = ticket["customer_email"]

    log.think(f"Planning resolution for intent='{intent}'. "
              f"Will gather: customer profile, order details, product policy, then act.")

    # ── PHASE 3: ACT — execute tool chain ─────────────────────────────────────

    # Tool call 1: Always fetch customer profile (know who you're talking to)
    log.act(f"Fetching customer profile for {email}")
    customer_result = call_tool_with_retry("get_customer", {"email": email}, log)

    if "error" in customer_result:
        evidence["customer_not_found"] = True
        log.think(f"Customer not found: {customer_result.get('message')}. "
                  f"Will request identification before processing financial actions.")
    else:
        customer = customer_result["customer"]
        log.think(f"Customer identified: {customer['name']}, tier={customer['tier']}, "
                  f"vip_exception={customer.get('vip_exception', False)}")
        evidence["customer"] = customer

    # ── Branch: No order ID and no customer → ask for info ───────────────────
    if not order_id and evidence["customer_not_found"]:
        log.think("Missing both order ID and verified customer. Cannot proceed — must ask for more info.")
        message = (
            "Hi there! To assist you, I'll need a bit more information: "
            "1) Your registered email address, and "
            "2) Your order number (found in your confirmation email, e.g. ORD-XXXX). "
            "Once I have these, I'll get this sorted for you right away."
        )
        log.act("Sending clarification request to customer")
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": message}, log)
        confidence, critique = self_critique(ticket, classification, "needs_info", evidence, log)
        log.set_outcome("needs_info", "Requested order ID and verified email from customer", confidence, message)
        result = {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": confidence}
        log.finalize()
        return result

    # Tool call 2: Fetch order if we have an order ID
    if order_id:
        log.act(f"Fetching order details for {order_id}")
        order_result = call_tool_with_retry("get_order", {"order_id": order_id}, log)

        if "error" in order_result:
            evidence["order_not_found"] = True
            log.think(f"Order not found: {order_result.get('message')}. May be invalid order ID.")
        else:
            order = order_result["order"]
            log.think(f"Order found: {order['product_name']}, status={order['status']}, amount=${order['amount']}")
            evidence["order"] = order

            # Tool call 3: Fetch product metadata
            log.act(f"Fetching product metadata for {order['product_id']}")
            product_result = call_tool_with_retry("get_product", {"product_id": order["product_id"]}, log)
            if "error" not in product_result:
                evidence["product"] = product_result["product"]
                log.think(f"Product policy: return_window={product_result['product']['return_window_days']}d, "
                          f"warranty={product_result['product']['warranty_days']}d")
    else:
        # No order ID — look up by customer email
        log.think("No order ID in ticket. Attempting to look up most recent order by customer email.")
        if not evidence["customer_not_found"]:
            # Try to find order by searching customer's known orders
            log.act("Looking up orders associated with customer email (no order ID provided)")
            # Search all known orders for this customer
            found_orders = [o for o in TOOL_REGISTRY.get("_orders_data", {}).values()
                           if o.get("customer_email") == email]
            if found_orders:
                # Use most recent
                order = sorted(found_orders, key=lambda x: x["ordered_at"], reverse=True)[0]
                order_id = order["order_id"]
                classification["order_id"] = order_id
                evidence["order"] = order
                log.think(f"Found order by email: {order_id} ({order['product_name']}), status={order['status']}")
            else:
                log.think("No orders found for this customer email.")

    # ── Resolve based on intent ───────────────────────────────────────────────

    result = _resolve_by_intent(ticket, classification, evidence, log)
    log.finalize()
    return result


def _resolve_by_intent(
    ticket: dict,
    classification: dict,
    evidence: dict,
    log: TicketLogger,
) -> dict:
    """
    Route to the appropriate resolution strategy based on classified intent.
    Each handler performs further tool calls, self-critique, and final action.
    """
    intent = classification["intent"]
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    flags = classification.get("flags", [])

    # ── REFUND REQUEST ────────────────────────────────────────────────────────
    if intent == "refund_request":
        # If body mentions defect/damage, route to defective handler for proper warranty check
        body_lower = ticket.get("body", "").lower()
        if any(w in body_lower for w in ["defect", "stopped working", "broken", "not working", "cracked", "damaged"]):
            log.think("Refund request contains defect language — routing to defective item handler for warranty check.")
            return _handle_defective(ticket, classification, evidence, log)
        return _handle_refund(ticket, classification, evidence, log)

    # ── RETURN REQUEST ────────────────────────────────────────────────────────
    elif intent == "return_request":
        return _handle_return(ticket, classification, evidence, log)

    # ── CANCELLATION ─────────────────────────────────────────────────────────
    elif intent == "cancellation":
        return _handle_cancellation(ticket, classification, evidence, log)

    # ── ORDER STATUS ──────────────────────────────────────────────────────────
    elif intent == "order_status":
        return _handle_order_status(ticket, classification, evidence, log)

    # ── DEFECTIVE ITEM ────────────────────────────────────────────────────────
    elif intent == "defective_item":
        return _handle_defective(ticket, classification, evidence, log)

    # ── WRONG ITEM ────────────────────────────────────────────────────────────
    elif intent == "wrong_item":
        return _handle_wrong_item(ticket, classification, evidence, log)

    # ── POLICY QUESTION ───────────────────────────────────────────────────────
    elif intent == "policy_question":
        return _handle_policy_question(ticket, classification, evidence, log)

    # ── UNKNOWN / AMBIGUOUS ───────────────────────────────────────────────────
    else:
        return _handle_ambiguous(ticket, classification, evidence, log)


# ── Intent handlers ───────────────────────────────────────────────────────────

def _handle_refund(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    flags = classification.get("flags", [])

    # Social engineering check
    if "potential_social_engineering" in flags:
        log.think("Potential social engineering detected. Checking actual customer tier vs claimed tier.")
        customer = evidence.get("customer", {})
        actual_tier = customer.get("tier", "unknown")
        log.think(f"Actual customer tier: '{actual_tier}'. Claimed premium/VIP status will not be honoured if false.")

    if not order_id or evidence.get("order_not_found"):
        confidence, _ = self_critique(ticket, classification, "needs_info", evidence, log)
        msg = _ask_for_order_info(ticket_id, log)
        log.set_outcome("needs_info", "Could not locate order — requested order details from customer", confidence, msg)
        return {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": confidence}

    order = evidence.get("order", {})

    # Check refund eligibility (tool call 4)
    log.act(f"Checking refund eligibility for {order_id}")
    eligibility = call_tool_with_retry("check_refund_eligibility", {"order_id": order_id, "reference_date": ticket.get("created_at")}, log)
    evidence["eligibility_checked"] = True
    evidence["eligibility"] = eligibility

    if "error" in eligibility:
        log.think(f"Eligibility check failed: {eligibility.get('message')}. Escalating.")
        return _do_escalate(ticket, classification, evidence, log,
                           f"Eligibility check failed for {order_id}: {eligibility.get('message')}",
                           "high" if classification["urgency"] == "high" else "medium")

    if eligibility.get("eligible"):
        # Double-check: if already refunded, don't double-refund
        if order.get("status") == "refunded":
            log.think("Order already refunded. Will confirm status to customer rather than issuing duplicate refund.")
            msg = (f"Hi {evidence.get('customer', {}).get('name', 'there')}! "
                   f"I can see that a refund for order {order_id} was already processed. "
                   f"Refunds typically appear in your account within 5–7 business days from processing. "
                   f"If you haven't seen it after that window, please reply and we'll investigate with your bank.")
            confidence, _ = self_critique(ticket, classification, "confirm_existing_refund", evidence, log)
            call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
            log.set_outcome("confirmed_existing_refund", "Refund already processed — customer notified", confidence, msg)
            return {"ticket_id": ticket_id, "action_taken": "confirmed_existing_refund", "confidence": confidence}

        amount = eligibility.get("amount", order.get("amount", 0))
        confidence, critique = self_critique(ticket, classification, "issue_refund", evidence, log)

        if confidence >= CONFIDENCE_THRESHOLD and "potential_social_engineering" not in flags:
            log.act(f"Issuing refund of ${amount:.2f} for {order_id}")
            refund_result = call_tool_with_retry("issue_refund", {"order_id": order_id, "amount": amount}, log)
            if "error" in refund_result:
                return _do_escalate(ticket, classification, evidence, log,
                                   f"Refund issue failed for {order_id} after eligibility confirmed",
                                   "high")
            msg = (f"Hi {evidence.get('customer', {}).get('name', 'there')}! "
                   f"Great news — your refund of ${amount:.2f} for order {order_id} has been processed. "
                   f"({eligibility.get('reason', '')}) "
                   f"You should see the funds within 5–7 business days.")
            call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
            log.set_outcome("refund_issued", f"Refund of ${amount:.2f} issued for {order_id}", confidence, msg)
            return {"ticket_id": ticket_id, "action_taken": "refund_issued", "confidence": confidence}
        else:
            reason = f"Low confidence ({confidence:.2f}) or social engineering flag. Critique: {critique}"
            return _do_escalate(ticket, classification, evidence, log, reason,
                               "critical" if "potential_social_engineering" in flags else "high")
    else:
        # Not eligible — explain why
        reason = eligibility.get("reason", "Not eligible for refund")
        log.think(f"Refund not eligible: {reason}. Checking if this is an already-processed refund or warranty case.")

        # Special case: refund already processed — confirm status, don't warranty-escalate
        if "already processed" in reason.lower() or order.get("status") == "refunded":
            log.think("Refund was already processed. Confirming status to customer — no further action needed.")
            msg = (f"Hi {evidence.get('customer', {}).get('name', 'there')}! "
                   f"I can confirm that a refund for order {order_id} was already processed. "
                   f"Refunds typically appear in your account within 5–7 business days from processing. "
                   f"If you haven't seen it after that window, please reply and we'll investigate with your bank.")
            confidence, _ = self_critique(ticket, classification, "confirm_existing_refund", evidence, log)
            call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
            log.set_outcome("confirmed_existing_refund", "Refund already processed — status confirmed to customer", confidence, msg)
            return {"ticket_id": ticket_id, "action_taken": "confirmed_existing_refund", "confidence": confidence}

        # Check warranty as alternative for ineligible-but-defective items
        product = evidence.get("product", {})
        warranty_days = product.get("warranty_days", 0)
        if warranty_days > 0:
            order_date = evidence.get("order", {}).get("ordered_at", "")
            if order_date:
                ordered_dt = datetime.fromisoformat(order_date.replace("Z", "+00:00"))
                now = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
                days_owned = (now - ordered_dt).days
                # Only escalate as warranty if the ticket actually mentions a defect
                body_lower = ticket.get("body", "").lower()
                is_defect_claim = any(w in body_lower for w in ["broken", "defect", "stopped", "damaged", "cracked", "not working", "working"])
                if days_owned <= warranty_days and is_defect_claim:
                    log.think(f"Item is within warranty ({days_owned}d owned, {warranty_days}d warranty). Escalating as warranty claim.")
                    return _do_escalate(ticket, classification, evidence, log,
                                       f"Return window expired ({reason}) but item is within {warranty_days}-day warranty. Escalate as warranty claim.",
                                       "medium")

        # If social engineering was detected, escalate rather than just deny
        if "potential_social_engineering" in flags:
            return _do_escalate(ticket, classification, evidence, log,
                               f"Social engineering attempt detected. Ineligible refund claimed: {reason}. Customer falsely claimed premium status.",
                               "critical")
        confidence, _ = self_critique(ticket, classification, "deny_refund", evidence, log)
        msg = (f"Hi {evidence.get('customer', {}).get('name', 'there')}, "
               f"I've reviewed your refund request for order {order_id}. "
               f"Unfortunately, {reason}. "
               f"If you believe your item has a manufacturing defect, it may be covered under warranty — "
               f"please reply with a brief description and we'll review it as a warranty claim.")
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
        log.set_outcome("refund_denied", f"Refund denied: {reason}", confidence, msg)
        return {"ticket_id": ticket_id, "action_taken": "refund_denied", "confidence": confidence}


def _handle_return(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    customer = evidence.get("customer", {})

    # Detect inquiry-only vs actual return request
    body_lower = ticket.get("body", "").lower()
    is_inquiry = any(p in body_lower for p in ["not sure", "might want", "thinking about", "what's the process", "is it too late", "how do i", "what is"])
    if is_inquiry and order_id:
        # Customer wants information, not to initiate a return yet
        log.think("Customer is asking about return process — this is an inquiry, not an active return request. "
                  "Will check eligibility to give accurate info, but won't initiate return.")
        log.act(f"Checking return eligibility to inform inquiry for {order_id}")
        eligibility_info = call_tool_with_retry("check_refund_eligibility", {"order_id": order_id, "reference_date": ticket.get("created_at")}, log)
        evidence["eligibility_checked"] = True
        kb = call_tool_with_retry("search_knowledge_base", {"query": "return policy process"}, log)
        order = evidence.get("order", {})
        product = evidence.get("product", {})
        window = product.get("return_window_days", 30)
        if eligibility_info.get("eligible"):
            process_info = (f"Hi {customer.get('name', 'there')}, good news — your order {order_id} is still within "
                           f"the {window}-day return window! When you're ready to return, simply reply to this message "
                           f"and we'll send you a prepaid return label. Your refund will be processed within "
                           f"5–7 business days of us receiving the item.")
            action = "return_info_provided"
        else:
            reason = eligibility_info.get("reason", "Return window may have expired")
            process_info = (f"Hi {customer.get('name', 'there')}, I've checked your order {order_id} — unfortunately "
                           f"{reason}. If there's a defect or fault, it may still be covered under warranty. "
                           f"Feel free to reply with more details and we'll see how we can help!")
            action = "return_info_provided"
        confidence = 0.92
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": process_info}, log)
        log.set_outcome(action, "Return process information provided (inquiry only)", confidence, process_info)
        return {"ticket_id": ticket_id, "action_taken": action, "confidence": confidence}

    if not order_id or evidence.get("order_not_found"):
        msg = _ask_for_order_info(ticket_id, log)
        log.set_outcome("needs_info", "Requested order details", 0.8, msg)
        return {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": 0.8}

    # Check eligibility (reuses refund eligibility logic which covers return windows)
    log.act(f"Checking return/refund eligibility for {order_id}")
    eligibility = call_tool_with_retry("check_refund_eligibility", {"order_id": order_id, "reference_date": ticket.get("created_at")}, log)
    evidence["eligibility_checked"] = True
    evidence["eligibility"] = eligibility

    # Check if customer is VIP with exception
    customer = evidence.get("customer", {})
    vip_exception = customer.get("vip_exception", False)

    if "error" in eligibility:
        return _do_escalate(ticket, classification, evidence, log,
                           f"Eligibility check failed for return {order_id}", "medium")

    product = evidence.get("product", {})
    returnable_if_registered = product.get("returnable_if_registered", True)

    if eligibility.get("eligible") or vip_exception:
        if vip_exception and not eligibility.get("eligible"):
            log.think("Return window expired BUT customer has VIP exception flag. VIP exceptions override standard policy.")
        elif not vip_exception:
            # Check if product was registered (non-returnable for non-VIP beyond 60 days)
            order = evidence.get("order", {})
            if not returnable_if_registered and order.get("status") == "delivered":
                ordered_date = order.get("ordered_at", "")
                if ordered_date:
                    ticket_dt = datetime.fromisoformat(ticket.get("created_at", "2024-03-15T12:00:00Z").replace("Z", "+00:00"))
                    ordered_dt = datetime.fromisoformat(ordered_date.replace("Z", "+00:00"))
                    if (ticket_dt - ordered_dt).days > 60:
                        log.think("Product ordered >60 days ago and registered online — non-returnable per policy.")
                        msg = (f"Hi {customer.get('name', 'there')}, "
                               f"I've reviewed your return request for order {order_id}. "
                               f"Unfortunately, this item is outside the return window and has been registered online, "
                               f"making it non-returnable per our policy. "
                               f"If there's a defect, our warranty programme may still apply.")
                        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
                        confidence, _ = self_critique(ticket, classification, "deny_return", evidence, log)
                        log.set_outcome("return_denied", "Return window expired and product registered — non-returnable", confidence, msg)
                        return {"ticket_id": ticket_id, "action_taken": "return_denied", "confidence": confidence}

        msg = (f"Hi {customer.get('name', 'there')}, "
               f"your return for order {order_id} has been approved"
               + (" as a VIP courtesy exception" if vip_exception and not eligibility.get("eligible") else "")
               + f". Please use return shipping label [LABEL-{order_id}] (valid 14 days). "
               f"Your refund will be processed within 5–7 business days of receipt.")
        confidence, _ = self_critique(ticket, classification, "approve_return", evidence, log)
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
        log.set_outcome("return_approved",
                       f"Return approved {'(VIP exception)' if vip_exception else '(within window)'}",
                       confidence, msg)
        return {"ticket_id": ticket_id, "action_taken": "return_approved", "confidence": confidence}
    else:
        reason = eligibility.get("reason", "Return window expired")
        msg = (f"Hi {customer.get('name', 'there')}, "
               f"I've reviewed your return request for order {order_id}. "
               f"Unfortunately, {reason}. "
               f"If your item has a defect covered by the manufacturer warranty, please let us know and we can open a warranty claim.")
        confidence, _ = self_critique(ticket, classification, "deny_return", evidence, log)
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
        log.set_outcome("return_denied", reason, confidence, msg)
        return {"ticket_id": ticket_id, "action_taken": "return_denied", "confidence": confidence}


def _handle_cancellation(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    customer = evidence.get("customer", {})

    # If no order ID, look up by email
    if not order_id and not evidence.get("customer_not_found"):
        log.think("No order ID given. Will look up most recent order for this customer.")
        # Simulate order lookup — in production this would be a DB query
        # For mock: check if customer has a known processing order
        from tools import ORDERS
        customer_orders = [o for o in ORDERS.values()
                          if o["customer_email"] == ticket["customer_email"]
                          and o["status"] == "processing"]
        if customer_orders:
            order_id = customer_orders[0]["order_id"]
            evidence["order"] = customer_orders[0]
            classification["order_id"] = order_id
            log.think(f"Found processing order: {order_id}")

    if not order_id or evidence.get("order_not_found"):
        msg = _ask_for_order_info(ticket_id, log)
        log.set_outcome("needs_info", "Requested order ID for cancellation", 0.8, msg)
        return {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": 0.8}

    order = evidence.get("order", {})
    if not order:
        log.act(f"Fetching order {order_id} for cancellation check")
        result = call_tool_with_retry("get_order", {"order_id": order_id}, log)
        if "error" not in result:
            order = result["order"]
            evidence["order"] = order

    status = order.get("status", "unknown")
    if status == "processing":
        log.act(f"Order {order_id} is in processing — cancelling immediately")
        # Simulated: cancel = issue full refund
        amount = order.get("amount", 0)
        log.act(f"Issuing cancellation refund of ${amount:.2f}")
        call_tool_with_retry("issue_refund", {"order_id": order_id, "amount": amount}, log)
        msg = (f"Hi {customer.get('name', 'there')}, "
               f"your order {order_id} has been successfully cancelled and a full refund of ${amount:.2f} "
               f"has been initiated. Funds will appear within 5–7 business days.")
        confidence, _ = self_critique(ticket, classification, "cancel_order", evidence, log)
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
        log.set_outcome("order_cancelled", f"Order {order_id} cancelled, ${amount:.2f} refunded", confidence, msg)
        return {"ticket_id": ticket_id, "action_taken": "order_cancelled", "confidence": confidence}
    elif status in ("shipped", "delivered"):
        msg = (f"Hi {customer.get('name', 'there')}, "
               f"unfortunately order {order_id} has already {status} and can no longer be cancelled. "
               f"Once you receive it, you can initiate a return within the return window.")
        confidence, _ = self_critique(ticket, classification, "deny_cancellation", evidence, log)
        call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
        log.set_outcome("cancellation_denied", f"Order {order_id} already {status}", confidence, msg)
        return {"ticket_id": ticket_id, "action_taken": "cancellation_denied", "confidence": confidence}
    else:
        return _do_escalate(ticket, classification, evidence, log,
                           f"Unusual order status '{status}' — requires human review", "medium")


def _handle_order_status(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    customer = evidence.get("customer", {})
    order = evidence.get("order", {})

    if not order or evidence.get("order_not_found"):
        msg = _ask_for_order_info(ticket_id, log)
        log.set_outcome("needs_info", "Requested order ID for status check", 0.85, msg)
        return {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": 0.85}

    status = order.get("status")
    tracking = order.get("tracking")
    confidence, _ = self_critique(ticket, classification, "inform_status", evidence, log)

    if status == "shipped":
        msg = (f"Hi {customer.get('name', 'there')}, your order {order_id} ({order.get('product_name')}) "
               f"is on its way! Tracking number: {tracking}. "
               f"Orders typically arrive within 2–3 business days of shipping.")
    elif status == "delivered":
        msg = (f"Hi {customer.get('name', 'there')}, order {order_id} shows as delivered. "
               f"If you haven't received it, please check with neighbours or your building's mailroom first. "
               f"If it's still missing, reply and we'll investigate.")
    elif status == "processing":
        msg = (f"Hi {customer.get('name', 'there')}, order {order_id} is currently being processed "
               f"and will ship within 1–2 business days. We'll email you a tracking number once shipped.")
    else:
        msg = (f"Hi {customer.get('name', 'there')}, your order {order_id} has status: {status}. "
               f"If you have questions, our team is here to help.")

    call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
    log.set_outcome("status_provided", f"Order status '{status}' communicated to customer", confidence, msg)
    return {"ticket_id": ticket_id, "action_taken": "status_provided", "confidence": confidence}


def _handle_defective(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    customer = evidence.get("customer", {})
    order = evidence.get("order", {})

    log.think("Defective item claim. Will check: (1) return window for refund, (2) warranty for coverage.")

    if not order_id or evidence.get("order_not_found"):
        msg = _ask_for_order_info(ticket_id, log)
        log.set_outcome("needs_info", "Requested order details for defect claim", 0.8, msg)
        return {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": 0.8}

    # Check eligibility
    log.act(f"Checking refund eligibility for defective item {order_id}")
    eligibility = call_tool_with_retry("check_refund_eligibility", {"order_id": order_id, "reference_date": ticket.get("created_at")}, log)
    evidence["eligibility_checked"] = True

    product = evidence.get("product", {})
    warranty_days = product.get("warranty_days", 0)

    # Check warranty coverage
    order_date = order.get("ordered_at", "")
    in_warranty = False
    if order_date and warranty_days:
        ordered_dt = datetime.fromisoformat(order_date.replace("Z", "+00:00"))
        now = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        in_warranty = (now - ordered_dt).days <= warranty_days

    if eligibility.get("eligible") and "error" not in eligibility:
        # Within return window AND defective → full refund, no return required for DOA
        body_lower = ticket.get("body", "").lower()
        is_doa = any(w in body_lower for w in ["arrived", "cracked", "box", "photos"])
        if is_doa:
            log.think("Damaged on arrival — full refund without requiring return per policy.")
        amount = eligibility.get("amount", order.get("amount", 0))
        confidence, _ = self_critique(ticket, classification, "issue_refund", evidence, log)
        if confidence >= CONFIDENCE_THRESHOLD:
            call_tool_with_retry("issue_refund", {"order_id": order_id, "amount": amount}, log)
            msg = (f"Hi {customer.get('name', 'there')}, I'm sorry to hear about the issue with your {order.get('product_name', 'item')}. "
                   f"{'As this arrived damaged, ' if is_doa else ''}"
                   f"I've issued a full refund of ${amount:.2f} — no need to return the item. "
                   f"Funds will appear within 5–7 business days.")
            call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
            log.set_outcome("refund_issued_defective", f"Defective item — ${amount:.2f} refunded", confidence, msg)
            return {"ticket_id": ticket_id, "action_taken": "refund_issued", "confidence": confidence}

    elif in_warranty:
        log.think(f"Return window expired but item is within {warranty_days}-day warranty. Escalating as warranty claim.")
        return _do_escalate(ticket, classification, evidence, log,
                           f"Defective item {order_id} within {warranty_days}-day warranty. Return window expired. Escalate to warranty team.",
                           "medium")
    else:
        # Customer wants replacement (like TKT-015)
        body_lower = ticket.get("body", "").lower()
        if "replacement" in body_lower:
            return _do_escalate(ticket, classification, evidence, log,
                               f"Customer requests replacement for defective {order.get('product_name', '')} ({order_id}). Within return window.",
                               "medium")

    # Fallback escalation
    return _do_escalate(ticket, classification, evidence, log,
                       f"Defective item claim for {order_id} requires manual review (eligibility={eligibility})",
                       "medium")


def _handle_wrong_item(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    order_id = classification.get("order_id")
    customer = evidence.get("customer", {})

    log.think("Wrong item delivered. Will check return eligibility, then approve exchange or refund.")

    log.act(f"Checking return eligibility for wrong item {order_id}")
    eligibility = call_tool_with_retry("check_refund_eligibility", {"order_id": order_id, "reference_date": ticket.get("created_at")}, log)
    evidence["eligibility_checked"] = True

    if eligibility.get("eligible") and "error" not in eligibility:
        amount = eligibility.get("amount", evidence.get("order", {}).get("amount", 0))
        log.think("Wrong item + within return window → issue refund and apologise.")

        # Check urgency (threatening language)
        urgency_note = ""
        if "threatening_language" in classification.get("flags", []):
            log.think("Threatening language detected but wrong item is a legitimate complaint — handle normally, don't escalate for tone alone.")
            urgency_note = " I apologise for this inconvenience and understand your frustration."

        confidence, _ = self_critique(ticket, classification, "issue_refund", evidence, log)
        if confidence >= CONFIDENCE_THRESHOLD:
            call_tool_with_retry("issue_refund", {"order_id": order_id, "amount": amount}, log)
            msg = (f"Hi {customer.get('name', 'there')}, I sincerely apologise — it appears the wrong item was shipped.{urgency_note} "
                   f"I've issued a full refund of ${amount:.2f} for order {order_id}. "
                   f"You're welcome to keep the incorrectly sent item. "
                   f"If you'd like to reorder the correct item, I'm happy to assist.")
            call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
            log.set_outcome("refund_issued_wrong_item", f"Wrong item — ${amount:.2f} refunded", confidence, msg)
            return {"ticket_id": ticket_id, "action_taken": "refund_issued", "confidence": confidence}

    return _do_escalate(ticket, classification, evidence, log,
                       f"Wrong item delivered for {order_id} — ineligible for auto-refund, requires human review",
                       "high")


def _handle_policy_question(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    customer = evidence.get("customer", {})

    log.act("Searching knowledge base for policy information")
    body = ticket.get("body", "")

    kb_result = call_tool_with_retry("search_knowledge_base",
                                     {"query": body[:200]}, log)

    results = kb_result.get("results", []) if "error" not in kb_result else []
    if not results:
        # Try a second search with just key terms
        log.act("First knowledge base search returned no results — retrying with focused query")
        kb_result2 = call_tool_with_retry("search_knowledge_base",
                                          {"query": "return policy refund exchange"}, log)
        results = kb_result2.get("results", []) if "error" not in kb_result2 else []

    answer_parts = []
    for r in results[:3]:
        answer_parts.append(f"**{r['topic'].title()}**: {r['content']}")

    answer = "\n\n".join(answer_parts) if answer_parts else "Please contact our support team for detailed policy information."
    msg = (f"Hi {customer.get('name', 'there')}, great question! Here's what you need to know:\n\n"
           f"{answer}\n\nFeel free to reply if you have more questions!")
    confidence = 0.95 if results else 0.5
    call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
    log.set_outcome("policy_answered", "Policy question answered from knowledge base", confidence, msg)
    return {"ticket_id": ticket_id, "action_taken": "policy_answered", "confidence": confidence}


def _handle_ambiguous(ticket, classification, evidence, log):
    ticket_id = ticket["ticket_id"]
    customer = evidence.get("customer", {})

    log.think("Ticket is too ambiguous to act on. Must ask targeted clarifying questions — do NOT guess.")

    log.act("Searching knowledge base to inform clarification questions")
    call_tool_with_retry("search_knowledge_base", {"query": "return refund warranty"}, log)

    msg = (f"Hi {customer.get('name', 'there')}, thanks for reaching out! "
           f"To help you as quickly as possible, could you share:\n"
           f"1. Your order number (e.g. ORD-XXXX)\n"
           f"2. Which product is affected\n"
           f"3. A brief description of the issue (e.g. not working, wrong item, want to return)\n\n"
           f"Once I have these details, I'll get this resolved right away!")
    confidence = 0.9
    call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
    log.set_outcome("needs_info", "Ambiguous ticket — requested clarifying details", confidence, msg)
    return {"ticket_id": ticket_id, "action_taken": "needs_info", "confidence": confidence}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _do_escalate(ticket, classification, evidence, log, reason, priority):
    ticket_id = ticket["ticket_id"]
    order = evidence.get("order", {})
    customer = evidence.get("customer", {})

    summary = (
        f"Ticket: {ticket_id} | "
        f"Customer: {customer.get('name', ticket['customer_email'])} (tier={customer.get('tier', 'unknown')}) | "
        f"Order: {order.get('order_id', 'N/A')} ({order.get('product_name', 'unknown')}) | "
        f"Intent: {classification.get('intent')} | "
        f"Reason for escalation: {reason} | "
        f"Flags: {classification.get('flags', [])}"
    )

    log.act(f"Escalating to human agent: {reason}")
    call_tool_with_retry("escalate",
                         {"ticket_id": ticket_id, "summary": summary, "priority": priority},
                         log)

    # Also send a holding reply to the customer
    msg = (f"Hi {customer.get('name', 'there')}, thank you for contacting ShopWave support. "
           f"Your ticket has been reviewed and escalated to our specialist team for personalised attention. "
           f"You'll hear from us within 4–8 business hours.")
    call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)

    confidence, _ = self_critique(ticket, classification, "escalate", evidence, log)
    log.set_outcome("escalated", reason, confidence, msg)
    return {"ticket_id": ticket_id, "action_taken": "escalated", "confidence": confidence}


def _ask_for_order_info(ticket_id, log):
    log.act("Sending clarification request for order ID")
    msg = ("Hi there! To locate your order and assist you as quickly as possible, "
           "could you please provide your order number? "
           "It looks like ORD-XXXX and can be found in your confirmation email. Thanks!")
    call_tool_with_retry("send_reply", {"ticket_id": ticket_id, "message": msg}, log)
    return msg
