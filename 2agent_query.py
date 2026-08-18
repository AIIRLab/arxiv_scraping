from __future__ import annotations

#================================================================================
#TWO-AGENT QUERY GENERATOR PIPELINE (target: 1000 verified rows)
#================================================================================


import argparse
import csv
import json
import logging
import os
import re
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
log = logging.getLogger("two_agent_gemma_pipeline")

csv.field_size_limit(sys.maxsize)

gemma_model = "google/gemma-2-9b-it"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are an expert in Information Retrieval.

Take the information from the paper as an input and generate a relevant query.

Rules:
1. Explicit Entity Anchoring: You must use concrete nouns, specific names, and clear subjects. Do not use vague pronouns like "he," "she," "it," "they," or "this company." The question must make complete sense if read entirely out of context.
2. Factually Bound: Every premise, fact, and assumption in your question must be 100 percent derived from the text. Do not introduce outside knowledge, real-world events, or hypothetical scenarios.
3. Natural Phrasing (No Boilerplate): Write like a human educator. Never start the question with robotic templates like "According to the text," "Based on the paragraph," "What does the author say about," or "In the provided text."
4. Conceptual Synthesizing: Target the relationship between ideas in the text. Avoid creating a question that can be answered by simply copy-pasting a single short phrase from the paragraph.
5. Conceal the Answer: Ensure the question asks for the core piece of information without accidentally revealing it, hinting at it, or answering itself.



Return ONLY valid JSON (no markdown, no extra text):
{
  "Paragraph": "...",
  "paragraph_id": "#####",
  "query": "...",
  "explanation": "Brief explanation of how the query is relevant to the text, no more than 1-2 sentences."
}"""

VERIFIER_SYSTEM_PROMPT = """You are a strict Information Retrieval query evaluator.

You will be given:
- "Paragraph": "...",
- "paragraph_id": "#####"
- "query": "...",
- "explanation": "Brief explanation of how the query is relevant to the text."


Your job is to decide whether the revised_query is relevant to the paragraph and a good question (TRUE), or if it is irrelevant, flawed, or nonsensical (FALSE).

[CRITERIA FOR TRUE]
To output TRUE, the question must meet ALL of the following conditions:
1. Factually Bound: The paragraph contains 100 percent of the information needed to answer the question. No outside knowledge or logical leaps are required.
2. Context Independent: The question has a clear, explicit subject and anchor entities. It can stand alone in a database without needing to see the paragraph to understand what is being asked.
3. Comprehension-Based: The question requires an understanding of the paragraph's meaning, rather than just matching a single exact string of copy-pasted text.

[CRITERIA FOR FALSE]
Output FALSE immediately if the question triggers ANY of these absolute failure states:
1. Self-Answering: The question accidentally reveals or includes its own core answer within the phrasing.
2. Speculative/Negative: The question asks about things completely absent from the text, or asks the reader to guess external possibilities.
3. Hallucination: The question introduces a new fact, name, date, or entity that does not exist in the source paragraph.
4. Premise Flaw: The question is built on a false setup, a logical fallacy, or a misinterpretation of the text.
5. Template-Dead: The question uses rigid, repetitive boilerplate language (e.g., "According to the text, what is...") instead of natural phrasing.

[OUTPUT FORMAT]
You must respond in this exact JSON format. Do not include any other text, markdown blocks, or commentary.
{
    "verdict": "FALSE",
    "reasoning": "A brief 1-2 sentence explanation mapping the question specifically to the criteria above."
}
or
{
    "verdict": "TRUE",
}"""


# ---------------------------------------------------------------------------
# Low-level generation helper
# ---------------------------------------------------------------------------

def _generate(
    system_prompt: str,
    user_message: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 768,
) -> str:
    """
    Runs a single-turn chat completion with the local Gemma model.
    """
    # Simple concatenation for Gemma
    full_prompt = f"{system_prompt}\n\n{user_message}\n\nResponse:"
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    # Move inputs to the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_len:]

    if generated_tokens.shape[-1] >= max_new_tokens:
        print(
            f"  [WARN] Generation hit max_new_tokens={max_new_tokens} — "
            f"output may be truncated mid-JSON. Consider raising max_new_tokens."
        )

    raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return raw.strip()

class JSONParseError(ValueError):
    """
    Raised only when the model produced output but it could not be parsed
    as JSON. Kept distinct from a bare ValueError so callers don't
    accidentally swallow unrelated errors (e.g. a crash inside
    model.generate()).
    """
    pass


def _parse_json(raw: str) -> dict:
    """
    Robustly extracts and parses the first *structurally valid* JSON object
    found in raw text (handles stray markdown fences / trailing commentary).
    """
    clean = re.sub(r"```(?:json)?", "", raw).strip()

    decoder = json.JSONDecoder()
    search_from = 0
    while True:
        start = clean.find("{", search_from)
        if start == -1:
            break
        try:
            obj, _end_index = decoder.raw_decode(clean, start)
            return obj
        except json.JSONDecodeError:
            search_from = start + 1

    raise JSONParseError(f"No valid JSON object found in model output:\n{raw}")


def agent1_generate_query(
    original_paragraph: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> dict:
    """
    Agent 1: Writes a query relevant to the given paragraph.

    Returns
    -------
    dict with keys:
        "Paragraph": "...",
        "paragraph_id": "#####",
        "query": "...",
        "explanation": "..."
    """
    raw = _generate(
        system_prompt=SYSTEM_PROMPT,
        user_message=f'Paragraph: "{original_paragraph}"',
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=768,
    )
    try:
        return _parse_json(raw)
    except JSONParseError:
        print(f"  [Injector] Raw model output that failed to parse:\n{raw}\n")
        raise


def agent2_verify_query(
    original_paragraph: str,
    revised_query: str,
    explanation: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> bool:
    """
    Agent 2: Independently verifies whether the rewritten query is a
    high-quality, relevant question based strictly on the content of
    the paragraph.

    Returns
    -------
    bool — True if the query is valid, False if it is flawed or irrelevant.
    """
    user_message = (
        f'Paragraph: "{original_paragraph}"\n'
        f'query: "{revised_query}"\n'
        f'explanation: "{explanation}"'
    )
    raw = _generate(
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_message=user_message,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=128,
    )
    try:
        result = _parse_json(raw)
    except JSONParseError:
        print(f"  [Verifier] Raw model output that failed to parse:\n{raw}\n")
        raise

    verdict = str(result.get("verdict", "FALSE")).strip().upper()
    is_good_query = verdict == "TRUE"

    if not is_good_query:
        reason = result.get("reasoning", "No reason provided.")
        print(f"  [Verifier] Rejected — {reason}")

    return is_good_query


def generate_question(
    original_paragraph: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_attempts: int = 3,
) -> dict | None:
    """
    Orchestrates Agent 1 (query generation) and Agent 2 (verification)
    for a single paragraph, retrying up to max_attempts times until a
    verified query is produced.

    Returns
    -------
    dict with keys "generated_query" and "explanation" on success,
    or None if no verified query was produced within max_attempts.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"\n  [Attempt {attempt}/{max_attempts}]")

        try:
            candidate = agent1_generate_query(
                original_paragraph=original_paragraph,
                tokenizer=tokenizer,
                model=model,
            )
        except JSONParseError:
            print("  [Injector] Failed to produce parseable JSON — retrying.")
            continue

        query = str(candidate.get("query", "")).strip()
        explanation = str(candidate.get("explanation", "")).strip()

        if not query:
            print("  [Injector] Empty query field — retrying.")
            continue

        try:
            is_valid = agent2_verify_query(
                original_paragraph=original_paragraph,
                revised_query=query,
                explanation=explanation,
                tokenizer=tokenizer,
                model=model,
            )
        except JSONParseError:
            print("  [Verifier] Failed to produce parseable JSON — retrying.")
            continue

        if is_valid:
            return {
                "generated_query": query,
                "explanation": explanation,
            }

    print(f"  [FAILED] No verified query after {max_attempts} attempts.")
    return None




# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_models(
    model_name: str = gemma_model,
    load_in_4bit: bool = False,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Loads the local Gemma tokenizer and model.
    """
    log.info("Loading model %s ...", model_name)

    # Set device explicitly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    quant_kwargs = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model without device_map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        **quant_kwargs,
    )
    
    # Move to device explicitly
    model = model.to(device)
    model.eval()
    log.info("Model loaded on %s.", device)

    return tokenizer, model


# ---------------------------------------------------------------------------
# JSON input helpers
# ---------------------------------------------------------------------------

def load_input_rows(input_path: str) -> list[dict]:
    """
    Reads a single paper JSON file and yields each paragraph separately 
    as an input row containing the paragraph_id and paragraph_text.
    """
    rows = []
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        
        # Extract the paragraphs dictionary safely
        paragraphs_dict = data.get("paragraphs", {})
        
        # Iterate over the paragraph IDs and their respective texts
        for paragraph_id, paragraph_text in paragraphs_dict.items():
            pid = str(paragraph_id or "").strip()
            text = str(paragraph_text or "").strip()
            
            # Skip rows if either the ID or the text content is empty
            if not pid or not text:
                continue
                
            rows.append({
                "paragraph_id": pid, 
                "paragraph_text": text
            })
            
    return rows

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> set[str]:
    """Returns the set of question_ids already consumed (accepted or
    exhausted-and-skipped) in a previous run, so a resumed run doesn't
    reprocess them."""
    if not os.path.exists(checkpoint_path):
        return set()
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed_question_ids", []))
    except (json.JSONDecodeError, OSError):
        log.warning("Could not read checkpoint file %s — starting fresh.", checkpoint_path)
        return set()


def save_checkpoint(checkpoint_path: str, processed_ids: set[str], accepted_count: int) -> None:
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_question_ids": sorted(processed_ids),
                "accepted_count": accepted_count,
            },
            f,
        )


# ---------------------------------------------------------------------------
# Main batch pipeline
# ---------------------------------------------------------------------------



log = logging.getLogger(__name__)

def run_pipeline(
    input_path: str,
    output_path: str,
    rejects_path: str,
    checkpoint_path: str,
    target_accepted: int,
    max_attempts: int,
    model_name: str,
    load_in_4bit: bool,
    resume: bool,
) -> None:
    tokenizer, model = load_models(model_name=model_name, load_in_4bit=load_in_4bit)

    all_rows = load_input_rows(input_path)
    log.info("Loaded %d paragraphs from %s", len(all_rows), input_path)

    processed_ids: set[str] = load_checkpoint(checkpoint_path) if resume else set()
    if processed_ids:
        log.info("Resuming — %d paragraph(s) already processed in a previous run.", len(processed_ids))

    # Helper to load existing list from JSON files safely if resuming
    def load_existing_json(file_path: str) -> list[dict]:
        if resume and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                log.warning("Could not parse %s as JSON. Starting fresh array.", file_path)
        return []

    # Initialize memory buffers with existing file records if resuming
    accepted_records = load_existing_json(output_path)
    rejected_records = load_existing_json(rejects_path)
    accepted_count = len(accepted_records)

    for row in all_rows:
        if accepted_count >= target_accepted:
            log.info("Reached target of %d verified paragraphs. Stopping.", target_accepted)
            break

        paragraph_id = row["paragraph_id"]
        og_paragraph = row["paragraph_text"]

        if paragraph_id in processed_ids:
            continue  # already handled in a previous run

        print("\n" + "=" * 60)
        print(f"[{accepted_count}/{target_accepted} accepted] PARAGRAPH_ID {paragraph_id}")
        print(f"ORIGINAL PARAGRAPH: {og_paragraph}")
        print("=" * 60)

        try:
            # Renamed function signature internally to align away from ambiguity terms
            result = generate_question(
                original_paragraph=og_paragraph,
                tokenizer=tokenizer,
                model=model,
                max_attempts=max_attempts,
            )
        except Exception as e:
            # A genuine crash (OOM, generation error, etc.) shouldn't break the whole job
            log.exception("Unhandled error while processing paragraph_id=%s: %s", paragraph_id, e)
            
            rejected_records.append({
                "paragraph_id": paragraph_id,
                "og_paragraph": og_paragraph,
                "reason": f"Unhandled error: {e}"
            })
            with open(rejects_path, "w", encoding="utf-8") as rejfile:
                json.dump(rejected_records, rejfile, indent=2, ensure_ascii=False)
                
            processed_ids.add(paragraph_id)
            save_checkpoint(checkpoint_path, processed_ids, accepted_count)
            continue

        processed_ids.add(paragraph_id)

        if result:
            print("\n[FINAL RESULT] Verified \u2713")
            print(f"  Generated query: {result['generated_query']}")
            print(f"  Explanation    : {result['explanation']}")
            
            accepted_records.append({
                "paragraph_id": paragraph_id,
                "og_paragraph": og_paragraph,
                "query": result["generated_query"],
                "explanation": result.get("explanation", ""),
            })

            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(accepted_records, outfile, indent=2, ensure_ascii=False)
                
            accepted_count += 1
        else:
            reason = f"No verified high-quality question generated after {max_attempts} attempts."
            print(f"[FINAL RESULT] Rejected — {reason}")
            
            rejected_records.append({
                "paragraph_id": paragraph_id,
                "og_paragraph": og_paragraph,
                "reason": reason
            })
            with open(rejects_path, "w", encoding="utf-8") as rejfile:
                json.dump(rejected_records, rejfile, indent=2, ensure_ascii=False)

        save_checkpoint(checkpoint_path, processed_ids, accepted_count)

    else:

        log.warning(
            "Exhausted all %d input paragraphs but only accepted %d/%d before stopping.",
            len(all_rows), accepted_count, target_accepted,
        )

    log.info("Done. %d verified high-quality questions written to %s", accepted_count, output_path)


# ---------------------------------------------------------------------------
# Test example — single paragraph, no JSON files required
# ---------------------------------------------------------------------------

def run_single_paragraph_test(paragraph: str | None = None, model_name: str = gemma_model) -> dict | None:
    # Default fallback uses a paragraph string matching your JSON structure
    test_paragraph = paragraph or (
        "The recent adoption of artificial intelligence in socio-technical systems raises concerns "
        "about the black-box nature of the resulting decisions in fields such as hiring, finance, "
        "admissions, etc. If data subjects—such as job applicants, loan applicants, and students—"
        "receive an unfavorable outcome, they may be interested in algorithmic recourse, which "
        "involves updating certain features to yield a more favorable result when re-evaluated by "
        "algorithmic decision-making. Unfortunately, when individuals do not fully understand the "
        "incremental steps needed to change their circumstances, they risk following misguided paths "
        "that can lead to significant, long-term adverse consequences. Existing recourse approaches "
        "focus exclusively on the final recourse goal but neglect the possible incremental steps to "
        "reach the goal with real-life constraints, user preferences, and model artifacts. To address "
        "this gap, we formulate a visual analytic workflow for incremental recourse planning in "
        "collaboration with AI/ML experts and contribute an interactive visualization interface that "
        "helps data subjects efficiently navigate the recourse alternatives and make an informed "
        "decision. We present a usage scenario and subjective feedback from observational studies "
        "with twelve graduate students using a real-world dataset, which demonstrates that our "
        "approach can be instrumental for data subjects in choosing a suitable recourse path."
    )

    tokenizer, model = load_models(model_name=model_name)

    print("\n" + "=" * 60)
    print(f"TEST PARAGRAPH: {test_paragraph}")
    print("=" * 60)

    result = generate_question(
        original_paragraph=test_paragraph,
        tokenizer=tokenizer,
        model=model,
        max_attempts=3,
    )

    print("\n" + "=" * 60)
    if result:
        print("[TEST RESULT: SUCCESS]")
        print(f"  Original paragraph : {test_paragraph}")
        print(f"  Generated query    : {result['generated_query']}")
        print(f"  Explanation        : {result['explanation']}")
    else:
        print("[TEST RESULT: FAILED] Pipeline could not produce a verified high-quality question.")
    print("=" * 60)

    return result



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-agent query-generation pipeline — stops once N verified questions are produced."
    )
    parser.add_argument(
        "--test",
        nargs="?",
        const="__default__",
        default=None,
        help=(
            "Run a single-paragraph smoke test instead of the full batch job. "
            "Optionally pass a paragraph string to test against; otherwise a "
            "built-in default paragraph is used."
        ),
    )
    parser.add_argument("--input", default="papers.json", help="Path to the input paper JSON file.")
    parser.add_argument("--output", default="gemma_verified.json", help="Path to write accepted rows to.")
    parser.add_argument("--rejects", default="gemma_rejected.json", help="Path to log rejected/failed rows to.")
    parser.add_argument("--checkpoint", default="gemma_checkpoint.json", help="Path to the resume checkpoint file.")
    parser.add_argument("--target", type=int, default=1000, help="Number of verified questions to stop at.")
    parser.add_argument("--max_attempts", type=int, default=3, help="Max Agent-1/Agent-2 retries per paragraph.")
    parser.add_argument(
        "--model_name",
        default=gemma_model,
        help="HuggingFace model ID for the local Gemma model used by both agents.",
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Load the model in 4-bit (bitsandbytes).")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore any existing checkpoint/output files and start completely fresh (overwrites output).",
    )
    args = parser.parse_args()

    if args.test is not None:
        paragraph = None if args.test == "__default__" else args.test
        run_single_paragraph_test(paragraph, model_name=args.model_name)
        return

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        rejects_path=args.rejects,
        checkpoint_path=args.checkpoint,
        target_accepted=args.target,
        max_attempts=args.max_attempts,
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()