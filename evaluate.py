"""
Evaluation script for the SHL Assessment Recommender Agent.
Simulates the 10 sample conversations (C1-C10) against our agent
and computes Recall@10 and behavior probes.
"""

import json
import os
import sys
import time
import requests
from typing import Optional

# API base URL
BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000")

# ─── Expected final recommendations (URLs) per conversation ────────────────
# Extracted from the sample conversations' final turn recommendations

EXPECTED_RECOMMENDATIONS = {
    "C1": [
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
        "https://www.shl.com/products/product-catalog/view/opq-leadership-report/",
    ],
    "C2": [
        "https://www.shl.com/products/product-catalog/view/smart-interview-live-coding/",
        "https://www.shl.com/products/product-catalog/view/linux-programming-general/",
        "https://www.shl.com/products/product-catalog/view/networking-and-implementation-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C3": [
        "https://www.shl.com/products/product-catalog/view/svar-spoken-english-us-new/",
        "https://www.shl.com/products/product-catalog/view/contact-center-call-simulation-new/",
        "https://www.shl.com/products/product-catalog/view/entry-level-customer-serv-retail-and-contact-center/",
        "https://www.shl.com/products/product-catalog/view/customer-service-phone-simulation/",
    ],
    "C4": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-numerical-reasoning/",
        "https://www.shl.com/products/product-catalog/view/financial-accounting-new/",
        "https://www.shl.com/products/product-catalog/view/basic-statistics-new/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C5": [
        "https://www.shl.com/products/product-catalog/view/global-skills-assessment/",
        "https://www.shl.com/products/product-catalog/view/global-skills-development-report/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "https://www.shl.com/products/product-catalog/view/opq-mq-sales-report/",
        "https://www.shl.com/products/product-catalog/view/salestransformationreport2-0-individualcontributor/",
    ],
    "C6": [
        "https://www.shl.com/products/product-catalog/view/safety-and-dependability-focus-8-0/",
        "https://www.shl.com/products/product-catalog/view/workplace-health-and-safety-new/",
    ],
    "C7": [
        "https://www.shl.com/products/product-catalog/view/hipaa-security/",
        "https://www.shl.com/products/product-catalog/view/medical-terminology-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-essentials-new/",
        "https://www.shl.com/products/product-catalog/view/dependability-and-safety-instrument-dsi/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C8": [
        "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
        "https://www.shl.com/products/product-catalog/view/microsoft-word-365-new/",
        "https://www.shl.com/products/product-catalog/view/ms-excel-new/",
        "https://www.shl.com/products/product-catalog/view/ms-word-new/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C9": [
        "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "https://www.shl.com/products/product-catalog/view/spring-new/",
        "https://www.shl.com/products/product-catalog/view/sql-new/",
        "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
        "https://www.shl.com/products/product-catalog/view/docker-new/",
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
    ],
    "C10": [
        "https://www.shl.com/products/product-catalog/view/shl-verify-interactive-g/",
        "https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
    ],
}

# ─── Conversation turns (user messages only, in order) ──────────────────────

CONVERSATIONS = {
    "C1": [
        "We need a solution for senior leadership.",
        "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
        "Selection — comparing candidates against a leadership benchmark.",
        "Perfect, that's what we need.",
    ],
    "C2": [
        "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
        "Yes, go ahead. Should I also add a cognitive test for this level?",
        "That works. Thanks.",
    ],
    "C3": [
        "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
        "English.",
        "US.",
        "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?",
        "Perfect — new simulation for volume, old solution for finalists. Confirmed.",
    ],
    "C4": [
        "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
        "Good. Can you also add a situational judgement element — work-context decision making for graduates?",
        "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates.",
    ],
    "C5": [
        "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
        "What's the difference between OPQ and OPQ MQ Sales Report?",
        "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.",
    ],
    "C6": [
        "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?",
        "What's the difference between the DSI and the Safety & Dependability 8.0?",
        "We're industrial. The 8.0 bundle is the right fit. Confirmed.",
    ],
    "C7": [
        "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?",
        "They're functionally bilingual — English fluent for written work. Go with the hybrid.",
        "Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?",
        "Understood. Keep the shortlist as-is.",
    ],
    "C8": [
        "I need to quickly screen admin assistants for Excel and Word daily.",
        "In that case, I am OK with adding a simulation - we want to capture the capabilities.",
        "That's good.",
    ],
    "C9": [
        'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required."',
        "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features.",
        "Senior IC. They lead design on their own services but don't manage other engineers directly.",
        "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.",
        "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
        "Do we really need Verify G+ on top of all the technical tests? Feels redundant.",
        "Keep Verify G+. Locking it in.",
    ],
    "C10": [
        "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
        "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.",
        "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
    ],
}


def call_chat(messages: list[dict]) -> dict:
    """Call the /chat endpoint with the given messages."""
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def simulate_conversation(conv_id: str) -> dict:
    """
    Simulate a multi-turn conversation.
    Returns the final response with accumulated history.
    """
    user_turns = CONVERSATIONS[conv_id]
    messages = []
    all_responses = []

    print(f"\n{'='*60}")
    print(f"Conversation {conv_id} ({len(user_turns)} user turns)")
    print(f"{'='*60}")

    for i, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        print(f"\n  [Turn {i+1}] USER: {user_msg[:80]}...")

        try:
            response = call_chat(messages)
        except Exception as e:
            print(f"  ERROR: {e}")
            response = {
                "reply": "Error occurred",
                "recommendations": None,
                "end_of_conversation": False,
            }

        reply = response.get("reply", "")
        recs = response.get("recommendations")
        eoc = response.get("end_of_conversation", False)

        print(f"  AGENT: {reply[:120]}...")
        if recs:
            print(f"  RECOMMENDATIONS ({len(recs)}):")
            for r in recs:
                print(f"    - {r['name']}")
        else:
            print(f"  RECOMMENDATIONS: null")
        print(f"  END_OF_CONVERSATION: {eoc}")

        # Add assistant response to history
        messages.append({"role": "assistant", "content": reply})
        all_responses.append(response)

        # Rate limit protection (Gemini 5 RPM => use >=12s between turns)
        delay_s = float(os.environ.get("EVAL_TURN_DELAY", "12"))
        time.sleep(delay_s)

    return {
        "conv_id": conv_id,
        "responses": all_responses,
        "final_response": all_responses[-1],
        "messages": messages,
    }


def compute_recall_at_k(predicted_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    """Compute Recall@K: fraction of expected URLs found in top-K predictions."""
    if not expected_urls:
        return 1.0
    predicted_set = set(predicted_urls[:k])
    expected_set = set(expected_urls)
    hits = predicted_set & expected_set
    return len(hits) / len(expected_set)


def evaluate_behavior_probes(results: dict) -> dict:
    """Evaluate behavior probes across all conversations."""
    probes = {}

    # Probe 1: Does agent clarify on vague first turn? (C1 turn 1 should have null recs)
    c1 = results.get("C1")
    if c1:
        first_resp = c1["responses"][0]
        probes["clarify_on_vague_query"] = first_resp.get("recommendations") is None
    else:
        probes["clarify_on_vague_query"] = None

    # Probe 2: Does agent refuse off-topic? (C7 turn 3 asks about legal/HIPAA compliance)
    c7 = results.get("C7")
    if c7 and len(c7["responses"]) >= 3:
        legal_resp = c7["responses"][2]
        reply = legal_resp.get("reply", "").lower()
        probes["refuses_legal_questions"] = any(
            phrase in reply
            for phrase in ["legal", "compliance", "cannot", "can't", "outside", "unable", "not able"]
        )
    else:
        probes["refuses_legal_questions"] = None

    # Probe 3: Does agent honor edits/refinements? (C9 turn 4 asks to add AWS/Docker, drop REST)
    c9 = results.get("C9")
    if c9 and len(c9["responses"]) >= 4:
        refined_resp = c9["responses"][3]
        recs = refined_resp.get("recommendations") or []
        rec_names = [r["name"].lower() for r in recs]
        rec_text = " ".join(rec_names)
        has_aws = any("aws" in name for name in rec_names)
        has_docker = any("docker" in name for name in rec_names)
        no_rest = not any("rest" in name for name in rec_names)
        probes["honors_refinements"] = has_aws and has_docker and no_rest
    else:
        probes["honors_refinements"] = None

    # Probe 4: Are all URLs from the catalog?
    catalog_path = os.path.join(os.path.dirname(__file__), "data", "catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    valid_urls = {item["url"] for item in catalog}

    all_urls_valid = True
    total_urls = 0
    invalid_urls = []
    for conv_id, result in results.items():
        for resp in result["responses"]:
            recs = resp.get("recommendations") or []
            for r in recs:
                total_urls += 1
                if r["url"] not in valid_urls:
                    all_urls_valid = False
                    invalid_urls.append(r["url"])
    probes["all_urls_from_catalog"] = all_urls_valid
    probes["total_urls_checked"] = total_urls
    probes["invalid_urls"] = invalid_urls

    # Probe 5: Schema always valid (all responses have reply, recommendations, end_of_conversation)
    schema_valid = True
    for conv_id, result in results.items():
        for resp in result["responses"]:
            if "reply" not in resp:
                schema_valid = False
            if "end_of_conversation" not in resp:
                schema_valid = False
    probes["schema_always_valid"] = schema_valid

    # Probe 6: end_of_conversation only on explicit confirmation
    eoc_correct = True
    for conv_id, result in results.items():
        for i, resp in enumerate(result["responses"]):
            if resp.get("end_of_conversation") and i < len(result["responses"]) - 1:
                # EOC set before the last turn - potentially premature
                eoc_correct = False
    probes["eoc_not_premature"] = eoc_correct

    return probes


def main():
    print("=" * 60)
    print("SHL Assessment Recommender — Evaluation")
    print(f"Target: {BASE_URL}")
    print("=" * 60)

    # Check health
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        print(f"Health check: {resp.json()}")
    except Exception as e:
        print(f"ERROR: Server not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    # Specify which conversations to run (or all)
    conv_ids = sys.argv[1:] if len(sys.argv) > 1 else list(CONVERSATIONS.keys())

    # Run evaluations
    results = {}
    recall_scores = {}

    for conv_id in conv_ids:
        if conv_id not in CONVERSATIONS:
            print(f"WARNING: Unknown conversation {conv_id}, skipping.")
            continue

        result = simulate_conversation(conv_id)
        results[conv_id] = result

        # Compute Recall@10
        final_recs = result["final_response"].get("recommendations") or []
        predicted_urls = [r["url"] for r in final_recs]
        expected_urls = EXPECTED_RECOMMENDATIONS.get(conv_id, [])
        recall = compute_recall_at_k(predicted_urls, expected_urls, k=10)
        recall_scores[conv_id] = recall

        print(f"\n  Recall@10 for {conv_id}: {recall:.2%} ({len(set(predicted_urls) & set(expected_urls))}/{len(expected_urls)} expected found)")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\n--- Recall@10 per Conversation ---")
    for conv_id, recall in recall_scores.items():
        expected = EXPECTED_RECOMMENDATIONS.get(conv_id, [])
        status = "OK" if recall >= 0.5 else "FAIL"
        print(f"  {status} {conv_id}: {recall:.2%} (expected {len(expected)} items)")

    avg_recall = sum(recall_scores.values()) / len(recall_scores) if recall_scores else 0
    print(f"\n  Average Recall@10: {avg_recall:.2%}")

    # Behavior probes
    probes = evaluate_behavior_probes(results)
    print("\n--- Behavior Probes ---")
    for probe, result in probes.items():
        if probe in ("invalid_urls", "total_urls_checked"):
            continue
        status = "OK" if result else "FAIL" if result is False else "UNKNOWN"
        print(f"  {status} {probe}: {result}")

    if probes.get("invalid_urls"):
        print(f"\n  Invalid URLs found:")
        for url in probes["invalid_urls"]:
            print(f"    - {url}")

    # Save results
    output = {
        "recall_scores": recall_scores,
        "average_recall": avg_recall,
        "behavior_probes": {
            k: v for k, v in probes.items() if k != "invalid_urls" or not v
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
