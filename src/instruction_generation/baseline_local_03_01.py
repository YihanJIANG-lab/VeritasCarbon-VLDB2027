"""
03-01 Baseline experiments using local Qwen2-72B-Instruct.

Three baselines — all use the SAME local model as CoDE for fair comparison:
  B1: Direct Prompting (single turn, simple prompt)
  B2: Self-Instruct style (seed templates + random selection)
  B3: WizardLM-style Evol-Instruct (iterative refinement)

Usage (standalone):
    python -m src.instruction_generation.baseline_local_03_01 \\
        --chunks_file data/processed_corpus/chunks_sampled_20000_by_year.jsonl \\
        --output_dir results/baselines \\
        --sample_size 500
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared local-model inference helper
# ---------------------------------------------------------------------------

class LocalModelInference:
    """Thin wrapper around a pre-loaded HuggingFace / Unsloth model."""

    def __init__(self, model, tokenizer, temperature: float = 0.7,
                 max_new_tokens: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        """Single-prompt inference."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
            )

        new_ids = [
            out[len(inp):]
            for inp, out in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(
            new_ids, skip_special_tokens=True
        )[0].strip()


# ---------------------------------------------------------------------------
# Response parser (shared across baselines)
# ---------------------------------------------------------------------------

def parse_instruction_response(text: str) -> Tuple[str, str]:
    """Parse model output into (instruction, answer).

    Supports both English and Chinese labels.
    """
    instruction = ""
    answer = ""

    lines = text.strip().split("\n")
    capture_answer = False
    answer_lines = []

    for line in lines:
        ll = line.strip()
        # Instruction line
        for prefix in ("Instruction:", "Instruction：", "指令:", "指令：",
                        "Question:", "Question：", "问题:", "问题："):
            if ll.startswith(prefix):
                instruction = ll[len(prefix):].strip()
                capture_answer = False
                break
        # Answer line
        for prefix in ("Answer:", "Answer：", "Response:", "Response：",
                        "回答:", "回答：", "答案:", "答案："):
            if ll.startswith(prefix):
                answer_lines = [ll[len(prefix):].strip()]
                capture_answer = True
                break
        else:
            if capture_answer:
                answer_lines.append(ll)

    answer = "\n".join(answer_lines).strip()

    if not instruction:
        instruction = "Analyze the following ESG-related content."
    if not answer or len(answer) < 10:
        answer = text.strip()

    return instruction, answer


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------

class LocalBaselineExperiment:
    """Three baseline methods, all powered by LOCAL Qwen2-72B."""

    SEED_INSTRUCTIONS = [
        "Summarize the main content of the following text.",
        "Generate a Q&A pair based on the following text.",
        "Extract key ESG information from the following text.",
        "Classify the ESG dimensions mentioned in the following text.",
        "Analyze the environmental management practices described.",
        "Identify carbon emission data and trends in the text.",
    ]

    def __init__(self, model, tokenizer, temperature: float = 0.7):
        self.llm = LocalModelInference(model, tokenizer, temperature=temperature)

    # ---- B1: Direct Prompting ------------------------------------------------

    def direct_prompting(self, chunk_text: str) -> Optional[Dict]:
        """B1 – single-turn, minimal prompt."""
        prompt = (
            "Based on the following ESG-related text, generate one high-quality "
            "instruction-answer pair.\n\n"
            f"Text:\n{chunk_text}\n\n"
            "Output format:\n"
            "Instruction: [instruction]\n"
            "Answer: [answer]"
        )
        try:
            output = self.llm.generate(prompt, max_new_tokens=1024)
            instr, ans = parse_instruction_response(output)
            return {
                "instruction": instr,
                "input": chunk_text,
                "output": ans,
                "method": "direct_prompting",
                "model": "Qwen2-72B-Instruct-Local",
            }
        except Exception as e:
            logger.error(f"direct_prompting failed: {e}")
            return None

    # ---- B2: Self-Instruct ---------------------------------------------------

    def self_instruct(self, chunk_text: str,
                      seed_instructions: Optional[List[str]] = None) -> Optional[Dict]:
        """B2 – Self-Instruct: sample seed template, then generate."""
        seeds = seed_instructions or self.SEED_INSTRUCTIONS
        selected_seed = random.choice(seeds)

        prompt = (
            f"Based on the text below, generate one instruction-answer pair.\n"
            f"Seed template: {selected_seed}\n\n"
            f"Text:\n{chunk_text}\n\n"
            "Output:\n"
            "1. One concrete instruction (based on the seed, tailored to the text)\n"
            "2. The corresponding answer\n\n"
            "Format:\n"
            "Instruction: [your instruction]\n"
            "Answer: [your answer]"
        )
        try:
            output = self.llm.generate(prompt, max_new_tokens=1024)
            instr, ans = parse_instruction_response(output)
            return {
                "instruction": instr,
                "input": chunk_text,
                "output": ans,
                "method": "self_instruct",
                "model": "Qwen2-72B-Instruct-Local",
                "seed_instruction": selected_seed,
            }
        except Exception as e:
            logger.error(f"self_instruct failed: {e}")
            return None

    # ---- B3: WizardLM-style Evol-Instruct ------------------------------------

    def wizardlm_evol(self, chunk_text: str,
                      evolution_rounds: int = 2) -> Optional[Dict]:
        """B3 – iterative instruction evolution, then answer generation."""
        # Step 1: initial instruction
        init_prompt = (
            "Based on the text below, generate an initial ESG-related instruction.\n\n"
            f"Text:\n{chunk_text}\n\n"
            "Output one initial instruction only."
        )
        try:
            instruction = self.llm.generate(init_prompt, max_new_tokens=512)

            # Step 2: evolve instruction
            for _ in range(evolution_rounds):
                evo_prompt = (
                    "Improve the following instruction to be more specific, "
                    "clear, and challenging.\n\n"
                    f"Current instruction: {instruction}\n\n"
                    f"Text:\n{chunk_text}\n\n"
                    "Output the improved instruction only."
                )
                instruction = self.llm.generate(evo_prompt, max_new_tokens=512)

            # Step 3: generate answer
            ans_prompt = (
                f"Instruction: {instruction}\n\n"
                f"Input: {chunk_text}\n\n"
                "Generate the response."
            )
            answer = self.llm.generate(ans_prompt, max_new_tokens=1024)

            return {
                "instruction": instruction.strip(),
                "input": chunk_text,
                "output": answer.strip(),
                "method": "wizardlm_evol",
                "model": "Qwen2-72B-Instruct-Local",
                "evolution_rounds": evolution_rounds,
            }
        except Exception as e:
            logger.error(f"wizardlm_evol failed: {e}")
            return None

    # ---- Runner --------------------------------------------------------------

    def run_all_baselines(
        self,
        chunks: List[Dict],
        output_dir: str = "results/baselines",
        sample_size: Optional[int] = None,
        methods: Optional[List[str]] = None,
        checkpoint_every: int = 50,
        resume: bool = True,
    ) -> Dict:
        """Run baseline comparison and save results.

        Args:
            chunks: list of chunk dicts (must have 'text', 'chunk_id')
            output_dir: output directory
            sample_size: randomly sample N chunks (None = all)
            methods: list of method names to run
            checkpoint_every: save intermediate results every N chunks
            resume: if True, load existing checkpoint and skip processed chunks

        Returns:
            dict of per-method stats
        """
        if methods is None:
            methods = ["direct_prompting", "self_instruct", "wizardlm_evol"]

        if sample_size and 0 < sample_size < len(chunks):
            # When resuming, use a fixed seed so the sample is reproducible
            rng = random.Random(42)
            chunks = rng.sample(chunks, sample_size)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        method_fn = {
            "direct_prompting": self.direct_prompting,
            "self_instruct": self.self_instruct,
            "wizardlm_evol": self.wizardlm_evol,
        }

        stats = {}
        for method in methods:
            fn = method_fn.get(method)
            if fn is None:
                logger.warning(f"Unknown method: {method}")
                continue

            output_file = out_path / f"{method}_results.jsonl"

            # ---- Resume: load existing checkpoint ----
            existing_results, done_ids = self._load_checkpoint(output_file) if resume else ([], set())
            if done_ids:
                logger.info(
                    f"[{method}] Resuming: found {len(existing_results)} existing results, "
                    f"{len(done_ids)} chunk IDs already processed"
                )

            remaining = [
                c for c in chunks if c.get("chunk_id", "") not in done_ids
            ]
            logger.info(
                f"Running baseline: {method} — "
                f"{len(remaining)} remaining / {len(chunks)} total chunks"
            )
            if not remaining:
                logger.info(f"[{method}] All chunks already processed, skipping.")
                stats[method] = {
                    "total": len(chunks),
                    "successful": len(existing_results),
                    "resumed_from": len(existing_results),
                    "newly_generated": 0,
                    "output_file": str(output_file),
                }
                continue

            results: List[Dict] = list(existing_results)   # start from checkpoint
            new_count = 0

            for idx, chunk in enumerate(
                tqdm(remaining, desc=f"Baseline ({method})")
            ):
                chunk_text = chunk.get("text", "")
                if not chunk_text or len(chunk_text) < 50:
                    continue

                result = fn(chunk_text)
                if result:
                    result["source_chunk_id"] = chunk.get("chunk_id", "")
                    result["source_doc_id"] = chunk.get("doc_id", "")
                    results.append(result)
                    new_count += 1

                # Checkpoint
                if checkpoint_every and (idx + 1) % checkpoint_every == 0:
                    self._save_jsonl(results, output_file)
                    logger.info(
                        f"  [{method}] checkpoint {idx+1}/{len(remaining)}, "
                        f"total_results={len(results)} (new={new_count})"
                    )

            self._save_jsonl(results, output_file)
            stats[method] = {
                "total": len(chunks),
                "successful": len(results),
                "resumed_from": len(existing_results),
                "newly_generated": new_count,
                "output_file": str(output_file),
            }
            logger.info(
                f"{method}: {len(results)}/{len(chunks)} succeeded "
                f"(resumed={len(existing_results)}, new={new_count})"
            )

        # Save summary
        summary_file = out_path / "baseline_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Summary saved: {summary_file}")

        return stats

    @staticmethod
    def _save_jsonl(records: List[Dict], filepath: Path):
        """Atomically save records to JSONL (write to tmp then rename)."""
        tmp = filepath.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.rename(filepath)

    @staticmethod
    def _load_checkpoint(filepath: Path):
        """Load existing results and return (records_list, set_of_chunk_ids)."""
        records: List[Dict] = []
        done_ids: set = set()
        if filepath.exists() and filepath.stat().st_size > 0:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        records.append(rec)
                        cid = rec.get("source_chunk_id", "")
                        if cid:
                            done_ids.add(cid)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {filepath}: {e}")
                records, done_ids = [], set()
        return records, done_ids


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run baseline experiments (local Qwen2-72B)")
    parser.add_argument("--chunks_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/baselines")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--model_path", type=str,
                        default="/hpc2hdd/home/yjiang909/models/Qwen2-72B-Instruct/Qwen/Qwen2-72B-Instruct")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Load chunks
    chunks = []
    with open(args.chunks_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} chunks from {args.chunks_file}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("Model loaded")

    experiment = LocalBaselineExperiment(model, tokenizer)
    experiment.run_all_baselines(
        chunks,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
    )
