"""
02-08 Ablation study framework.

Expert count, collaboration mode, feedback iterations, domain knowledge.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .instruction_generator_02_05 import InstructionGenerator

logger = logging.getLogger(__name__)


class AblationStudy:
    """Ablation study runner for CoDE components."""
    def __init__(self, base_config: Dict):
        self.base_config = base_config
    def run_expert_count_ablation(
        self,
        chunks: List[Dict],
        expert_counts: List[int] = [1, 2, 5],
        output_dir: str = "results/ablation/expert_count",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Ablation: expert count (1 vs 2 vs 5)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if sample_size and sample_size > 0:
            import random
            if len(chunks) > sample_size:
                chunks = random.sample(chunks, sample_size)
        results = {}
        for count in expert_counts:
            logger.info(f"Expert count ablation: {count}")
            config = self.base_config.copy()
            config["coe"]["min_experts"] = count
            config["coe"]["max_experts"] = count
            generator = InstructionGenerator(config_path=None, project_root=None)
            generator.config = config
            generator.expert_selector.min_experts = count
            generator.expert_selector.max_experts = count
            output_file = output_path / f"expert_count_{count}.jsonl"
            stats = generator.generate_batch(chunks, str(output_file))
            results[f"expert_count_{count}"] = {
                "config": {"expert_count": count},
                "stats": stats,
                "output_file": str(output_file)
            }
        return results
    def run_collaboration_ablation(
        self,
        chunks: List[Dict],
        collaboration_modes: List[str] = ["none", "sequential", "parallel"],
        output_dir: str = "results/ablation/collaboration",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Ablation: collaboration (none / sequential / parallel)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if sample_size and sample_size > 0:
            import random
            if len(chunks) > sample_size:
                chunks = random.sample(chunks, sample_size)
        results = {}
        for mode in collaboration_modes:
            logger.info(f"Collaboration ablation: {mode}")
            config = self.base_config.copy()
            if mode == "none":
                config["coe"]["enable_collaboration"] = False
            else:
                config["coe"]["enable_collaboration"] = True
                config["coe"]["collaboration_mode"] = mode
            generator = InstructionGenerator(config_path=None, project_root=None)
            generator.config = config
            generator.coe_framework.enable_collaboration = config["coe"]["enable_collaboration"]
            generator.coe_framework.collaboration_mode = config["coe"].get("collaboration_mode", "sequential")
            output_file = output_path / f"collaboration_{mode}.jsonl"
            stats = generator.generate_batch(chunks, str(output_file))
            results[f"collaboration_{mode}"] = {
                "config": {"collaboration_mode": mode},
                "stats": stats,
                "output_file": str(output_file)
            }
        return results
    def run_feedback_ablation(
        self,
        chunks: List[Dict],
        max_iterations_list: List[int] = [0, 1, 2],
        output_dir: str = "results/ablation/feedback",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Ablation: feedback iterations (0 vs 1 vs 2)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if sample_size and sample_size > 0:
            import random
            if len(chunks) > sample_size:
                chunks = random.sample(chunks, sample_size)
        results = {}
        for max_iter in max_iterations_list:
            logger.info(f"Feedback ablation: max_iter={max_iter}")
            config = self.base_config.copy()
            config["coe"]["max_iterations"] = max_iter
            generator = InstructionGenerator(config_path=None, project_root=None)
            generator.config = config
            generator.coe_framework.max_iterations = max_iter
            output_file = output_path / f"feedback_{max_iter}.jsonl"
            stats = generator.generate_batch(chunks, str(output_file))
            results[f"feedback_{max_iter}"] = {
                "config": {"max_iterations": max_iter},
                "stats": stats,
                "output_file": str(output_file)
            }
        return results
    def run_knowledge_ablation(
        self,
        chunks: List[Dict],
        use_knowledge_list: List[bool] = [False, True],
        output_dir: str = "results/ablation/knowledge",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Ablation: domain knowledge on/off."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if sample_size and sample_size > 0:
            import random
            if len(chunks) > sample_size:
                chunks = random.sample(chunks, sample_size)
        results = {}
        for use_knowledge in use_knowledge_list:
            logger.info(f"Knowledge ablation: use_knowledge={use_knowledge}")
            generator = InstructionGenerator(config_path=None, project_root=None)
            output_file = output_path / f"knowledge_{use_knowledge}.jsonl"
            stats = {"total_chunks": len(chunks), "processed": 0, "successful": 0, "failed": 0}
            with open(output_file, "w", encoding="utf-8") as f_out:
                for chunk in tqdm(chunks, desc=f"Generate (knowledge={use_knowledge})"):
                    result = generator.generate_for_chunk(
                        chunk, use_knowledge=use_knowledge, use_adaptive_selection=True
                    )
                    stats["processed"] += 1
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stats["successful"] += 1
                    else:
                        stats["failed"] += 1
            results[f"knowledge_{use_knowledge}"] = {
                "config": {"use_knowledge": use_knowledge},
                "stats": stats,
                "output_file": str(output_file)
            }
        return results
    def run_all_ablation_studies(
        self,
        chunks: List[Dict],
        output_dir: str = "results/ablation",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Run all ablation studies; save summary to ablation_summary.json."""
        all_results = {}
        logger.info("="*60)
        logger.info("Expert count ablation")
        logger.info("="*60)
        all_results["expert_count"] = self.run_expert_count_ablation(
            chunks, output_dir=f"{output_dir}/expert_count", sample_size=sample_size
        )
        logger.info("="*60)
        logger.info("Collaboration ablation")
        logger.info("="*60)
        all_results["collaboration"] = self.run_collaboration_ablation(
            chunks, output_dir=f"{output_dir}/collaboration", sample_size=sample_size
        )
        logger.info("="*60)
        logger.info("Feedback ablation")
        logger.info("="*60)
        all_results["feedback"] = self.run_feedback_ablation(
            chunks, output_dir=f"{output_dir}/feedback", sample_size=sample_size
        )
        logger.info("="*60)
        logger.info("Knowledge ablation")
        logger.info("="*60)
        all_results["knowledge"] = self.run_knowledge_ablation(
            chunks, output_dir=f"{output_dir}/knowledge", sample_size=sample_size
        )
        summary_file = Path(output_dir) / "ablation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"All ablations done; summary: {summary_file}")
        return all_results

