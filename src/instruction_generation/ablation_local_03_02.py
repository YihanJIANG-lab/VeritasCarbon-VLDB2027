"""
03-02 Ablation studies using local Qwen2-72B-Instruct.

Four ablation dimensions:
  A1: Expert count  (1 / 2 / 3 / 5)
  A2: Collaboration mode  (none / sequential / parallel)
  A3: Feedback iterations  (0 / 1 / 2)
  A4: Domain knowledge injection  (on / off)

Each ablation re-instantiates the CoDE pipeline with modified config,
then generates QA pairs on **the same** chunk sample for fair comparison.

The local Qwen2-72B model and tokenizer must be passed in from notebook.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


class LocalAblationStudy:
    """Ablation study runner that uses a LOCAL model (no API calls)."""

    def __init__(
        self,
        model,
        tokenizer,
        experts: Dict,
        expert_selector,
        knowledge_injector,
        coe_framework_cls,
        expert_output_cls,
        meta_expert=None,
        topic_extractor=None,
        temperature: float = 0.7,
    ):
        """
        Args:
            model, tokenizer: pre-loaded Qwen2-72B
            experts: dict name -> expert instance (LocalQwenExpertAdapter subclass)
            expert_selector: ExpertSelector instance
            knowledge_injector: DomainKnowledgeInjector instance
            coe_framework_cls: COEFramework class (for re-instantiation)
            expert_output_cls: ExpertOutput dataclass
            meta_expert: optional MetaExpert (local) instance
            topic_extractor: optional TopicExtractor (local) instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.experts = experts
        self.expert_selector = expert_selector
        self.knowledge_injector = knowledge_injector
        self.COEFramework = coe_framework_cls
        self.ExpertOutput = expert_output_cls
        self.meta_expert = meta_expert
        self.topic_extractor = topic_extractor
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Core: generate one QA pair with given config overrides
    # ------------------------------------------------------------------

    def _generate_for_chunk(
        self,
        chunk: Dict,
        coe: object,
        selector,
        use_knowledge: bool = True,
        use_meta_expert: bool = True,
    ) -> Optional[Dict]:
        """Generate an instruction-answer pair for one chunk using given CoDE config."""
        chunk_text = chunk.get("text", "")
        if not chunk_text or len(chunk_text) < 50:
            return None

        try:
            # 1. Expert selection
            selected_experts, selection_reasons = selector.select_experts(chunk_text)
            if not selected_experts:
                return None

            # 2. Topic extraction (if meta-expert enabled)
            core_topics = []
            dynamic_instructions = {}
            if use_meta_expert and self.topic_extractor:
                try:
                    core_topics = self.topic_extractor.extract_topics(chunk_text, max_topics=5)
                except Exception:
                    pass
            if use_meta_expert and self.meta_expert:
                try:
                    for expert_name in selected_experts:
                        instruction = self.meta_expert.generate_instruction(
                            expert_type=expert_name,
                            chunk_text=chunk_text,
                            core_topics=core_topics,
                            section_name=None,
                            all_experts=selected_experts,
                        )
                        dynamic_instructions[expert_name] = instruction
                except Exception:
                    pass

            # 3. Domain knowledge
            context = {}
            if use_knowledge:
                knowledge_items = self.knowledge_injector.retrieve_knowledge(chunk_text)
                context["knowledge_items"] = knowledge_items

            if dynamic_instructions:
                context["dynamic_instructions"] = dynamic_instructions

            # 4. CoDE generation
            output = coe.generate(
                chunk_text=chunk_text,
                selected_experts=selected_experts,
                context=context,
            )

            return {
                "instruction": output.instruction,
                "input": chunk_text,
                "output": output.response,
                "source_chunk_id": chunk.get("chunk_id", ""),
                "metadata": {
                    "experts": selected_experts,
                    "quality_score": output.quality_score,
                    "core_topics": core_topics,
                    "model": "Qwen2-72B-Instruct-Local",
                },
            }
        except Exception as e:
            logger.error(f"Ablation generation failed: {e}")
            return None

    def _run_one_condition(
        self,
        label: str,
        chunks: List[Dict],
        coe,
        selector,
        use_knowledge: bool = True,
        use_meta_expert: bool = True,
        output_file: Optional[Path] = None,
        checkpoint_every: int = 30,
        resume: bool = True,
    ) -> Dict:
        """Run one ablation condition with checkpoint + resume support.

        Args:
            checkpoint_every: save intermediate results every N chunks
            resume: if True, load existing file and skip processed chunk IDs
        """
        # ---- Resume: load existing checkpoint ----
        existing_results, done_ids = (
            self._load_checkpoint(output_file) if (resume and output_file) else ([], set())
        )
        if done_ids:
            logger.info(
                f"[{label}] Resuming: {len(existing_results)} existing results, "
                f"{len(done_ids)} chunk IDs done"
            )

        remaining = [c for c in chunks if c.get("chunk_id", "") not in done_ids]
        logger.info(
            f"[{label}] {len(remaining)} remaining / {len(chunks)} total chunks"
        )

        results: List[Dict] = list(existing_results)
        new_count = 0

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(tqdm(remaining, desc=f"Ablation ({label})")):
            r = self._generate_for_chunk(
                chunk, coe, selector,
                use_knowledge=use_knowledge,
                use_meta_expert=use_meta_expert,
            )
            if r:
                r["ablation_label"] = label
                results.append(r)
                new_count += 1

            # ---- Periodic checkpoint ----
            if (
                output_file
                and checkpoint_every
                and (idx + 1) % checkpoint_every == 0
            ):
                self._save_jsonl(results, output_file)
                logger.info(
                    f"  [{label}] checkpoint {idx+1}/{len(remaining)}, "
                    f"total={len(results)} (new={new_count})"
                )

        # Final save
        if output_file:
            self._save_jsonl(results, output_file)

        quality_scores = [
            r["metadata"]["quality_score"]
            for r in results
            if r["metadata"].get("quality_score", 0) > 0
        ]

        return {
            "label": label,
            "total": len(chunks),
            "successful": len(results),
            "resumed_from": len(existing_results),
            "newly_generated": new_count,
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "output_file": str(output_file) if output_file else None,
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

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
        if filepath and filepath.exists() and filepath.stat().st_size > 0:
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

    # ------------------------------------------------------------------
    # A1: Expert count
    # ------------------------------------------------------------------

    def run_expert_count_ablation(
        self,
        chunks: List[Dict],
        expert_counts: List[int] = [1, 2, 3, 5],
        output_dir: str = "results/ablation/expert_count",
    ) -> Dict:
        results = {}
        for count in expert_counts:
            label = f"expert_count_{count}"
            # Clone selector with overridden count
            sel = self._clone_selector(min_experts=count, max_experts=count)
            coe = self.COEFramework(
                experts=self.experts,
                enable_collaboration=(count > 1),
                collaboration_mode="sequential",
                max_iterations=2,
                quality_threshold=0.7,
            )
            stats = self._run_one_condition(
                label, chunks, coe, sel,
                output_file=Path(output_dir) / f"{label}.jsonl",
            )
            results[label] = stats
            logger.info(f"  {label}: {stats['successful']}/{stats['total']}, "
                        f"avg_q={stats['avg_quality']:.4f}")
        return results

    # ------------------------------------------------------------------
    # A2: Collaboration mode
    # ------------------------------------------------------------------

    def run_collaboration_ablation(
        self,
        chunks: List[Dict],
        modes: List[str] = ["none", "sequential", "parallel"],
        output_dir: str = "results/ablation/collaboration",
    ) -> Dict:
        results = {}
        sel = self._clone_selector(min_experts=2, max_experts=3)
        for mode in modes:
            label = f"collab_{mode}"
            enable = mode != "none"
            coe = self.COEFramework(
                experts=self.experts,
                enable_collaboration=enable,
                collaboration_mode=mode if enable else "sequential",
                max_iterations=2,
                quality_threshold=0.7,
            )
            stats = self._run_one_condition(
                label, chunks, coe, sel,
                output_file=Path(output_dir) / f"{label}.jsonl",
            )
            results[label] = stats
            logger.info(f"  {label}: {stats['successful']}/{stats['total']}, "
                        f"avg_q={stats['avg_quality']:.4f}")
        return results

    # ------------------------------------------------------------------
    # A3: Feedback iterations
    # ------------------------------------------------------------------

    def run_feedback_ablation(
        self,
        chunks: List[Dict],
        iteration_counts: List[int] = [0, 1, 2],
        output_dir: str = "results/ablation/feedback",
    ) -> Dict:
        """Ablation on feedback rounds R.

        Semantics:
          R=0  →  1 generation pass, NO refinement feedback
          R=1  →  1 generation pass + 1 feedback refinement round
          R=2  →  1 generation pass + 2 feedback refinement rounds

        The COEFramework's ``max_iterations`` controls the total number of
        passes through the expert chain.  Therefore the mapping is:

            max_iterations = R + 1   (initial generation + R refinements)

        Bug-fix (2026-03): previous code passed ``max_iterations = R``
        directly, causing R=0 to skip generation entirely (all-zero output).
        """
        results = {}
        sel = self._clone_selector(min_experts=2, max_experts=3)
        for feedback_rounds in iteration_counts:
            label = f"feedback_{feedback_rounds}"
            # +1: always perform at least the initial generation pass
            coe = self.COEFramework(
                experts=self.experts,
                enable_collaboration=True,
                collaboration_mode="sequential",
                max_iterations=feedback_rounds + 1,
                quality_threshold=0.7,
            )
            stats = self._run_one_condition(
                label, chunks, coe, sel,
                output_file=Path(output_dir) / f"{label}.jsonl",
            )
            results[label] = stats
            logger.info(f"  {label}: {stats['successful']}/{stats['total']}, "
                        f"avg_q={stats['avg_quality']:.4f}")
        return results

    # ------------------------------------------------------------------
    # A4: Domain knowledge injection
    # ------------------------------------------------------------------

    def run_knowledge_ablation(
        self,
        chunks: List[Dict],
        output_dir: str = "results/ablation/knowledge",
    ) -> Dict:
        results = {}
        sel = self._clone_selector()
        coe = self.COEFramework(
            experts=self.experts,
            enable_collaboration=True,
            collaboration_mode="sequential",
            max_iterations=2,
            quality_threshold=0.7,
        )
        for use_k in [False, True]:
            label = f"knowledge_{'on' if use_k else 'off'}"
            stats = self._run_one_condition(
                label, chunks, coe, sel,
                use_knowledge=use_k,
                output_file=Path(output_dir) / f"{label}.jsonl",
            )
            results[label] = stats
            logger.info(f"  {label}: {stats['successful']}/{stats['total']}, "
                        f"avg_q={stats['avg_quality']:.4f}")
        return results

    # ------------------------------------------------------------------
    # Run all four ablations
    # ------------------------------------------------------------------

    def run_all(
        self,
        chunks: List[Dict],
        output_dir: str = "results/ablation",
        sample_size: int = 200,
    ) -> Dict:
        """Run all 4 ablation studies on a fixed sample of chunks."""
        if sample_size and 0 < sample_size < len(chunks):
            chunks = random.sample(chunks, sample_size)
            logger.info(f"Sampled {sample_size} chunks for ablation")

        all_results = {}

        logger.info("=" * 60)
        logger.info("A1: Expert count ablation")
        logger.info("=" * 60)
        all_results["expert_count"] = self.run_expert_count_ablation(
            chunks, output_dir=f"{output_dir}/expert_count"
        )

        logger.info("=" * 60)
        logger.info("A2: Collaboration mode ablation")
        logger.info("=" * 60)
        all_results["collaboration"] = self.run_collaboration_ablation(
            chunks, output_dir=f"{output_dir}/collaboration"
        )

        logger.info("=" * 60)
        logger.info("A3: Feedback iterations ablation")
        logger.info("=" * 60)
        all_results["feedback"] = self.run_feedback_ablation(
            chunks, output_dir=f"{output_dir}/feedback"
        )

        logger.info("=" * 60)
        logger.info("A4: Knowledge injection ablation")
        logger.info("=" * 60)
        all_results["knowledge"] = self.run_knowledge_ablation(
            chunks, output_dir=f"{output_dir}/knowledge"
        )

        # Save summary
        summary_file = Path(output_dir) / "ablation_summary.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"All ablation results saved: {summary_file}")

        return all_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clone_selector(self, min_experts=None, max_experts=None):
        """Clone the ExpertSelector with overridden expert count."""
        from .expert_selector_02_01 import ExpertSelector

        sel = ExpertSelector(
            use_ml_classifier=False,
            min_experts=min_experts or self.expert_selector.min_experts,
            max_experts=max_experts or self.expert_selector.max_experts,
            use_layered_selection=self.expert_selector.use_layered_selection,
            layered_config={
                "base_layer_required": self.expert_selector.base_layer_required,
                "analysis_layer_threshold": self.expert_selector.analysis_layer_threshold,
                "verification_layer_threshold": self.expert_selector.verification_layer_threshold,
                "graph_layer_threshold": self.expert_selector.graph_layer_threshold,
            },
        )
        return sel
