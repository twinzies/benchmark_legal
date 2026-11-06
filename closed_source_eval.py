import argparse
import csv
import json
import logging
import os
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Set

from anthropic import Anthropic

# import ipdb
from google import genai
from openai import OpenAI
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "closed_source_eval.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# model arg -> (provider, api_model_name)
MODEL_CONFIG = {
    "GPT-5.5":        ("openai","gpt-5.5-2026-04-23"),
    "o4":             ("openai","o4-mini-2025-04-16"),
    "Gemini-flash":   ("google","gemini-3.1-flash-lite"),
    "Gemini-pro":     ("google","gemini-3.1-pro-preview"),
    "Claude-Opus":    ("anthropic","claude-opus-4-7"),
    "SaulLM-7B":      ("vllm","Equall/Saul-7B-Instruct-v1"), # Ignore for paper, only to validate the vLLM setup.
    "SaulLM-141B":    ("vllm","Equall/SaulLM-141B-Instruct"),
    # Base model of SaulLM-141B (the paper's "Mixtral-141B-Instruct") - isolates
    # what SaulLM's legal continued-pretraining adds. Served via OpenRouter. Interestingly, scores a higher weighted-F1 than SaulLM !
    "Mixtral-141B":   ("vllm","mistralai/mixtral-8x22b-instruct"),
    "GLM-5.2":        ("vllm","z-ai/glm-5.2"),
    # The current best open Qwen
    "Qwen3.6-35B":    ("vllm","qwen/qwen3.6-35b-a3b"),
}

# -----------------------------
# Step 0: Helper functions
# -----------------------------
def read_json(path: str | Path) -> List[Dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: List[Dict], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_env(env_path: str | Path) -> None:
    path = Path(env_path)
    if not path.exists():
        raise FileNotFoundError(f".env file not found at: {path}")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

PROGRESS_CSV_COLUMNS = ["case_id", "model_name"]

def load_progress(progress_csv: Path, model_name: str) -> Set[str]:
    """Return case_ids already successfully processed for this model."""
    if not progress_csv.exists():
        return set()
    with progress_csv.open(newline="", encoding="utf-8") as f:
        return {
            row["case_id"]
            for row in csv.DictReader(f)
            if row.get("model_name") == model_name
        }

def mark_progress(progress_csv: Path, case_id: str, model_name: str) -> None:
    """Append one completed case to the progress CSV."""
    write_header = not progress_csv.exists()
    with progress_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PROGRESS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({"case_id": case_id, "model_name": model_name})

def build_client(provider: str, model_key: str | None = None) -> Any:
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "google":
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    if provider == "anthropic":
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider == "vllm":
        # Each self-hosted model gets its own env vars, derived from its
        # MODEL_CONFIG key: e.g. "SaulLM-7B" -> SAULLM_7B_BASE_URL / SAULLM_7B_API_KEY
        prefix = model_key.upper().replace("-", "_").replace(".", "_")
        return OpenAI(
            api_key=os.environ[f"{prefix}_API_KEY"],
            base_url=os.environ[f"{prefix}_BASE_URL"],
        )
    raise ValueError(f"Unknown provider: {provider}")

# -----------------------------
# Step 1: Generate predictions using LLM
# -----------------------------

def build_evaluation_prompt_vanilla(facts: str, claims: str) -> str:
    return f"""
You are a UK legal assistant.

Your task is to read the facts and claims of a court case, perform legal analysis, and predict all applicable UK statutes that are relevant to resolving the legal issues raised by the case facts and claims.

Instructions:
1. Identify all UK statutes that are applicable based on the facts and claims.
2. Predict statutes in the standard canonical form of the UK law being referred to.
   Examples:
   - "Consumer Rights Act 2015"
   - "Sale of Goods Act 1979"
   - "Companies Act 2006"
3. Include Acts, Regulations, Rules, Orders, and other UK legislation where relevant.
4. Do not explain your reasoning.
5. If no applicable UK statute can be identified, output an empty list.
6. Return only the required JSON object.

Facts:
\"\"\"
{facts}
\"\"\"

Claims:
\"\"\"
{claims}
\"\"\"
""".strip()

# For Gemini - note that descriptions become type hints.
class PredictionResult(BaseModel):
    uk_statute_references_predicted: List[str]

def predict_applicable_law_once(
    facts: str,
    claims: str,
    client: Any,
    model_name: str,
    provider: str,
    max_retries: int = 3,
) -> dict:
    """
    Predict applicable UK statutes from case facts and claims.
    Returns:
    {
        "uk_statute_references_predicted": [...]
    }
    """
    eval_prompt = build_evaluation_prompt_vanilla(facts=facts, claims=claims)
    schema = {
        "type": "object",
        "properties": {
            "uk_statute_references_predicted": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["uk_statute_references_predicted"],
        "additionalProperties": False,
    }
    system_msg = "You are a careful UK legal assistant that predicts applicable UK legislation from case facts and claims."

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    system=system_msg,
                    messages=[{"role": "user", "content": eval_prompt}],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": schema,
                        }
                    },
                )

                logging.info(f"Anthropic RAW response: {response}") 
                content = response.content[0].text

            elif provider == "google":
                response = client.models.generate_content(
                    model=model_name,
                    contents=eval_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_msg,
                        response_mime_type="application/json",
                        response_schema=PredictionResult.model_json_schema(),
                    ),
                )
                logging.info(f"Gemini RAW response: {response}") 
                content = response.text

            else:
                # Mistral-based models (e.g. SaulLM) don't support a separate
                # system role: merge the system message into the user prompt.
                if provider == "vllm":
                    messages = [
                        {"role": "user", "content": f"{system_msg}\n\n{eval_prompt}"}
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": eval_prompt},
                    ]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "applicable_law_prediction",
                            "schema": schema,
                            "strict": True,
                        },
                    }
                )
                logging.info(f"OpenAI RAW response: {response}")
                content = response.choices[0].message.content

            logging.info(f"LLM {model_name} response: {content}")
            parsed = json.loads(content)
            predicted = parsed.get("uk_statute_references_predicted", [])
            if not isinstance(predicted, list):
                raise ValueError("uk_statute_references_predicted must be a list")
            predicted = [x.strip() for x in predicted if isinstance(x, str) and x.strip()]
            return {"uk_statute_references_predicted": predicted}

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                sleep(2 * attempt)
            else:
                raise last_error


def predict_applicable_law_all(
    test_json_path: str | Path,
    output_dir: str | Path,
    model_name: str,
    provider: str,
    api_client: Any,
    max_retries: int = 3,
) -> str:

    test_json_path = Path(test_json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
    output_file = output_dir / f"applicable_law_predictions_{safe_model_name}.json"
    progress_csv = output_dir / "progress.csv"

    cases = read_json(test_json_path)
    if not isinstance(cases, list):
        raise ValueError("Input JSON must contain a list of cases.")

    # Load already-done case_ids and their saved predictions
    done_ids = load_progress(progress_csv, model_name)
    existing_predictions: Dict[str, Dict] = {}
    if output_file.exists() and done_ids:
        for p in read_json(output_file):
            existing_predictions[p["case_id"]] = p
        logging.info(f"Resuming: {len(done_ids)} cases already done, skipping them.")

    predictions: List[Dict[str, Any]] = []

    with tqdm(total=len(cases), desc=f"{model_name}", unit="case") as pbar:
        for case in cases:
            case_id = case.get("case_id", "")
            case_name = case.get("case_name", "")

            if case_id in done_ids:
                predictions.append(existing_predictions[case_id])
                pbar.update(1)
                continue

            facts = case.get("facts", "")
            claims = case.get("claims", "")

            try:
                prediction = predict_applicable_law_once(
                    facts=facts,
                    claims=claims,
                    client=api_client,
                    model_name=model_name,
                    provider=provider,
                    max_retries=max_retries,
                )
                predicted_statutes = prediction.get("uk_statute_references_predicted", [])
                mark_progress(progress_csv, case_id, model_name)
                predictions.append({
                    "case_id": case_id,
                    "case_name": case_name,
                    "facts": facts,
                    "claims": claims,
                    "uk_statute_references_predicted": predicted_statutes,
                })
            except Exception as e:
                logging.error(f"ERROR on case_id={case_id} ({case_name}): {e} — omitting from output.")

            pbar.update(1)

    write_json(predictions, output_file)
    logging.info(f"Done. Wrote {len(predictions)} predictions to:\n {output_file}")
    return str(output_file)

# -----------------------------
# Step 2: Evaluate predictions
# -----------------------------

def normalize_label_list(labels: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for x in labels:
        if not isinstance(x, str):
            continue
        if not x:
            continue
        x = x.strip().lower()
        if x not in seen:
            cleaned.append(x)
            seen.add(x)
    return cleaned


def evaluate_applicable_law_predictions(
    prediction_json_path: str | Path,
    gold_json_path: str | Path,
    model_name: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    pred_cases = read_json(prediction_json_path)
    gold_cases = read_json(gold_json_path)

    pred_map = {case["case_id"]: case for case in pred_cases}
    gold_map = {case["case_id"]: case for case in gold_cases}

    y_true = []
    y_pred = []
    missing_predictions = []

    for case_id, gold_case in gold_map.items():
        if case_id not in pred_map:
            missing_predictions.append(case_id) # Is this intended behavior for Error cases?
            continue
        gold_labels = normalize_label_list(gold_case.get("uk_statute_references_normalized", []))
        logging.info(f"case_id={case_id} gold_labels={gold_labels}")

        pred_labels = normalize_label_list(pred_map[case_id].get("uk_statute_references_predicted", []))
        logging.info(f"case_id={case_id} pred_labels={pred_labels}")

        y_true.append(gold_labels)
        y_pred.append(pred_labels)

    all_labels = sorted(list(set(label for labels in (y_true + y_pred) for label in labels)))
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(y_true)
    Y_pred = mlb.transform(y_pred)

    metrics = {
        "num_cases_evaluated": len(y_true),
        "num_unique_labels": len(all_labels),
        "subset_accuracy": accuracy_score(Y_true, Y_pred),
        "micro_precision": precision_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(Y_true, Y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(Y_true, Y_pred, average="micro", zero_division=0),
        "macro_precision": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(Y_true, Y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(Y_true, Y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(Y_true, Y_pred, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(Y_true, Y_pred),
        "samples_jaccard": jaccard_score(Y_true, Y_pred, average="samples", zero_division=0),
    }

    if verbose:
        print("\n==============================")
        print("Applicable Law Prediction Evaluation")
        print("==============================")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        if missing_predictions:
            print(f"\nWARNING: Missing predictions for case_id(s): {len(missing_predictions)}.")
        print("\n==============================")
        print("Per-label Classification Report")
        print("==============================")
        print(classification_report(Y_true, Y_pred, target_names=mlb.classes_, zero_division=0, digits=4))

    safe_model_name = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
    results_output_path = Path(prediction_json_path).parent / f"evaluation_results_{safe_model_name}.json"
    write_json({"metrics": metrics, "missing_predictions": missing_predictions}, results_output_path)

    return metrics


# -----------------------------
# CLI entrypoint
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Closed-source LLM evaluation for applicable law prediction.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIG.keys()),
        help="Model to evaluate with.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["test-run", "full-run"],
        help="'test-run' uses sample_cases_applicable_law.json; 'full-run' uses test_cases_applicable_law.json.",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script's location
    script_dir = Path(__file__).parent
    load_env(script_dir / ".env")

    provider, api_model_name = MODEL_CONFIG[args.model]
    client = build_client(provider, model_key=args.model)

    data_file = (
        "sample_cases_applicable_law.json" if args.mode == "test-run"
        else "test_cases_applicable_law.json"
    )
    data_path = script_dir / "data" / data_file
    pred_dir = script_dir / "predictions"

    print(f"Model:    {args.model} → {api_model_name} (provider: {provider})")
    print(f"Mode:     {args.mode} → {data_path.name}")

    # Step 1: predict
    pred_file_path = predict_applicable_law_all(
        test_json_path=data_path,
        output_dir=pred_dir,
        model_name=api_model_name,
        provider=provider,
        api_client=client,
    )
    logging.info(f"Saved predictions to {pred_file_path}")

    # Step 2: evaluate
    logging.info("Running evaluation.")
    metrics = evaluate_applicable_law_predictions(
        prediction_json_path=pred_file_path,
        gold_json_path=data_path,
        model_name=api_model_name,
        verbose=True,
    )
    print(metrics)
