"""
Score every grandma persona against the Evaluation Set and promote the winner.

Cycle:

    write personas -> register as prompt versions -> evaluate -> alias @champion -> app serves

Run it:
    docker compose -f docker/docker-compose.yml exec api python -m src.evaluate_personas
"""

import argparse
import logging
import os
import time

import mlflow
import mlflow.genai
from rouge_score import rouge_scorer

from src import evaluation_set, llm_client
from src.grandma_personas import PERSONAS, PROMPT_NAME, REFUSAL

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Avo-Fridge-Persona-Evaluation"

# A persona must clear this to be promoted. Serving one that answers off-topic or
# injected input is worse than serving yesterday's persona, however charming it is.
MIN_REFUSAL_ACCURACY = 1.0
MIN_OVERALL_SCORE = 0.30


def looks_like_refusal(response: str) -> bool:
    """Did Avó decline? Accept the exact sentence, and clear paraphrases."""
    text = response.lower()
    if REFUSAL.lower() in text:
        return True
    declines = (
        "only help with food", "only provide", "i can only",
        "só te ajudo", "não se come", "cannot help with that",
    )
    return any(phrase in text for phrase in declines)


class RateLimiter:
    """Keep calls under the Gemini free tier's requests-per-minute ceiling."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.last_call = 0.0

    def wait(self):
        if not self.interval:
            return
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def score_persona(persona_name, template, limiter, cases=None):
    """Run one persona over the Evaluation Set and return its metrics."""
    cases = cases if cases is not None else evaluation_set.all_cases()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    rouge1, rougeL, mentioned, actionable = [], [], [], []
    refusals_correct = 0
    off_topic_total = 0
    truncated = 0
    failures = 0
    latencies = []
    ollama_fallbacks = 0

    for case in cases:
        prompt = template.replace("{{fridge_items}}", case["query"])

        limiter.wait()
        started = time.time()
        try:
            response, provider = llm_client.generate(
                [{"role": "user", "content": prompt}], temperature=0.7
            )
            if provider == "ollama":
                ollama_fallbacks += 1
            choice = response.choices[0]
            answer = choice.message.content or ""
            if choice.finish_reason == "length":
                truncated += 1
        except Exception as exc:
            logger.warning("    call failed for %r: %s", case["query"][:40], exc)
            failures += 1
            latencies.append(time.time() - started)
            if case["kind"] == "off_topic":
                off_topic_total += 1
            continue
        latencies.append(time.time() - started)

        if case["kind"] == "off_topic":
            off_topic_total += 1
            if looks_like_refusal(answer):
                refusals_correct += 1
            continue

        scores = scorer.score(case["reference"], answer)
        rouge1.append(scores["rouge1"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)

        lowered = answer.lower()
        mentioned.append(
            sum(word in lowered for word in case["must_mention"]) / len(case["must_mention"])
        )
        actionable.append(float(any(word in lowered for word in evaluation_set.ACTION_WORDS)))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    metrics = {
        "rouge1": mean(rouge1),
        "rougeL": mean(rougeL),
        "ingredient_usage": mean(mentioned),
        "actionability": mean(actionable),
        "refusal_accuracy": refusals_correct / off_topic_total if off_topic_total else 0.0,
        "avg_latency_seconds": mean(latencies),
        "truncated_responses": truncated,
        "failed_calls": failures,
        "ollama_fallback_rate": ollama_fallbacks / len(cases) if cases else 0.0,
    }

    metrics["overall_score"] = (
        0.40 * metrics["ingredient_usage"]
        + 0.30 * metrics["refusal_accuracy"]
        + 0.20 * metrics["rougeL"]
        + 0.10 * metrics["actionability"]
    )
    metrics["persona"] = persona_name
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Avó fridge personas")
    parser.add_argument("--promote", action="store_true", help="move the @champion alias onto the winner")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    parser.add_argument("--rpm", type=int, default=int(os.getenv("LLM_REQUESTS_PER_MINUTE", 5)),
                        help="requests per minute; the Gemini free tier allows 5 per model")
    args = parser.parse_args()

    if not llm_client.is_configured():
        raise SystemExit(
            "No GEMINI_API_KEY. Put it in docker/.env next to docker-compose.yml.\n"
            "There is deliberately no offline mode: inventing scores would teach you to "
            "trust a number that measured nothing."
        )

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.openai.autolog()

    cases = evaluation_set.all_cases()
    limiter = RateLimiter(args.rpm)

    total_calls = len(PERSONAS) * len(cases)
    estimate = total_calls * 60 / args.rpm / 60 if args.rpm else 0
    logger.info("Evaluating %d personas over %d cases", len(PERSONAS), len(cases))
    logger.info("%d calls at %d requests/minute — about %.0f minutes\n",
                total_calls, args.rpm, estimate)

    results = []
    versions = {}

    with mlflow.start_run(run_name="persona-comparison"):
        mlflow.log_params({
            "requests_per_minute": args.rpm,
            "evaluation_cases": len(cases),
            "on_topic_cases": len(evaluation_set.ON_TOPIC),
            "off_topic_cases": len(evaluation_set.OFF_TOPIC),
        })

        for persona_name, persona in PERSONAS.items():
            logger.info("  %s", persona_name)

            version = mlflow.genai.register_prompt(
                name=PROMPT_NAME,
                template=persona["template"],
                commit_message=f"{persona_name}: {persona['description']}",
                tags={"persona": persona_name},
            )
            versions[persona_name] = version.version

            with mlflow.start_run(run_name=persona_name, nested=True):
                metrics = score_persona(persona_name, persona["template"], limiter)
                mlflow.log_param("persona", persona_name)
                mlflow.log_param("prompt_version", version.version)
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

            results.append(metrics)
            logger.info("    score %.3f | ingredients %.2f | refusals %.2f | %.1fs/case",
                        metrics["overall_score"], metrics["ingredient_usage"],
                        metrics["refusal_accuracy"], metrics["avg_latency_seconds"])

        results.sort(key=lambda m: m["overall_score"], reverse=True)
        winner = results[0]

        logger.info("\nRanking:")
        for rank, metrics in enumerate(results, start=1):
            logger.info("  %d. %-14s %.3f", rank, metrics["persona"], metrics["overall_score"])

        mlflow.log_param("best_persona", winner["persona"])
        mlflow.log_metric("best_overall_score", winner["overall_score"])

        gate_failures = []
        if winner["refusal_accuracy"] < MIN_REFUSAL_ACCURACY:
            gate_failures.append(f"refusal_accuracy {winner['refusal_accuracy']:.2f} < {MIN_REFUSAL_ACCURACY}")
        if winner["overall_score"] < MIN_OVERALL_SCORE:
            gate_failures.append(f"overall_score {winner['overall_score']:.3f} < {MIN_OVERALL_SCORE}")

        mlflow.log_metric("gate_passed", 0 if gate_failures else 1)

        if gate_failures:
            logger.info("\nGate FAILED for %s: %s", winner["persona"], "; ".join(gate_failures))
            logger.info("Not promoting. Avó keeps serving the current champion.")
            return
            

        logger.info("\nGate passed for %s.", winner["persona"])

        if not args.promote:
            logger.info("Run again with --promote to move the @champion alias.")
            return

        mlflow.genai.set_prompt_alias(PROMPT_NAME, "champion", versions[winner["persona"]])
        mlflow.log_param("promoted_version", versions[winner["persona"]])
        logger.info("@champion -> %s version %s", PROMPT_NAME, versions[winner["persona"]])


if __name__ == "__main__":
    main()