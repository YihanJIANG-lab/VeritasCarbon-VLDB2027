"""
02-03 CoDE (Council of Domain Experts) framework.

Expert collaboration, feedback loop, and quality evaluation.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ExpertOutput:
    """Expert generation output."""
    expert_name: str
    instruction: str
    response: str
    quality_score: float
    metadata: Dict[str, Any]


class COEFramework:
    """Council of Domain Experts (CoDE): coordinates multiple expert agents to generate instruction-answer pairs."""

    def __init__(
        self,
        experts: Dict[str, Any],
        enable_collaboration: bool = True,
        collaboration_mode: str = "sequential",
        max_iterations: int = 2,
        quality_threshold: float = 0.7
    ):
        """
        Args:
            experts: Dict of expert_name -> expert_instance
            enable_collaboration: Whether to enable collaboration
            collaboration_mode: "sequential" or "parallel"
            max_iterations: Max feedback loop iterations
            quality_threshold: Quality score threshold
        """
        self.experts = experts
        self.enable_collaboration = enable_collaboration
        self.collaboration_mode = collaboration_mode
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    def generate_sequential(
        self,
        chunk_text: str,
        selected_experts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ExpertOutput:
        """Sequential collaboration: each expert sees the previous expert's output."""
        previous_output = None
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"Collaboration round {iteration}")
            for i, expert_name in enumerate(selected_experts):
                if expert_name not in self.experts:
                    logger.warning(f"Expert {expert_name} not found, skipping")
                    continue
                expert = self.experts[expert_name]
                if previous_output and i > 0:
                    input_text = f"""
Previous expert output:
Instruction: {previous_output.instruction}
Response: {previous_output.response}

Generate higher-quality instruction-answer pairs based on the above.
Original chunk:
{chunk_text}
"""
                else:
                    input_text = chunk_text
                
                try:
                    expert_context = context.copy() if context else {}
                    if context and "dynamic_instructions" in context:
                        dynamic_instructions = context.get("dynamic_instructions", {})
                        if expert_name in dynamic_instructions:
                            expert_context["dynamic_instruction"] = dynamic_instructions[expert_name]
                        elif "dynamic_instruction" in context:
                            expert_context["dynamic_instruction"] = context["dynamic_instruction"]
                    output = expert.generate(input_text, context=expert_context)
                    quality_score = self._evaluate_quality(output, chunk_text)
                    output.quality_score = quality_score
                    if quality_score >= self.quality_threshold and iteration > 0:
                        logger.debug("Quality above threshold; stopping iteration")
                        return output
                    previous_output = output
                except Exception as e:
                    logger.error(f"Expert {expert_name} generation failed: {e}")
                    continue
            if previous_output and previous_output.quality_score < self.quality_threshold:
                logger.debug(f"Quality below threshold ({previous_output.quality_score:.2f} < {self.quality_threshold}), continuing")
            else:
                break
        
        return previous_output if previous_output else ExpertOutput(
            expert_name="unknown",
            instruction="",
            response="",
            quality_score=0.0,
            metadata={}
        )
    
    def generate_parallel(
        self,
        chunk_text: str,
        selected_experts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ExpertOutput:
        """Parallel collaboration: all experts generate, then best output is chosen."""
        outputs = []
        for expert_name in selected_experts:
            if expert_name not in self.experts:
                continue
            expert = self.experts[expert_name]
            try:
                expert_context = context.copy() if context else {}
                if context and "dynamic_instructions" in context:
                    dynamic_instructions = context.get("dynamic_instructions", {})
                    if expert_name in dynamic_instructions:
                        expert_context["dynamic_instruction"] = dynamic_instructions[expert_name]
                    elif "dynamic_instruction" in context:
                        expert_context["dynamic_instruction"] = context["dynamic_instruction"]
                output = expert.generate(chunk_text, context=expert_context)
                quality_score = self._evaluate_quality(output, chunk_text)
                output.quality_score = quality_score
                outputs.append(output)
            except Exception as e:
                logger.error(f"Expert {expert_name} generation failed: {e}")
                continue
        if not outputs:
            return ExpertOutput(
                expert_name="unknown",
                instruction="",
                response="",
                quality_score=0.0,
                metadata={}
            )
        
        best_output = max(outputs, key=lambda x: x.quality_score)
        return best_output
    
    def generate(
        self,
        chunk_text: str,
        selected_experts: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> ExpertOutput:
        """
        Generate instruction-answer pair (main entry).
        Args:
            chunk_text: Chunk text
            selected_experts: Selected expert IDs
            context: Context (e.g. domain knowledge)
        Returns:
            ExpertOutput
        """
        if not self.enable_collaboration or len(selected_experts) == 1:
            expert_name = selected_experts[0]
            if expert_name in self.experts:
                expert = self.experts[expert_name]
                expert_context = context.copy() if context else {}
                if context and "dynamic_instructions" in context:
                    dynamic_instructions = context.get("dynamic_instructions", {})
                    if expert_name in dynamic_instructions:
                        expert_context["dynamic_instruction"] = dynamic_instructions[expert_name]
                    elif "dynamic_instruction" in context:
                        expert_context["dynamic_instruction"] = context["dynamic_instruction"]
                
                output = expert.generate(chunk_text, context=expert_context)
                quality_score = self._evaluate_quality(output, chunk_text)
                output.quality_score = quality_score
                return output
        
        if self.collaboration_mode == "sequential":
            return self.generate_sequential(chunk_text, selected_experts, context)
        else:
            return self.generate_parallel(chunk_text, selected_experts, context)
    
    def _evaluate_quality(
        self,
        output: ExpertOutput,
        chunk_text: str
    ) -> float:
        """Evaluate output quality (0-1). Supports CJK via character n-grams."""
        score = 0.0
        if len(output.instruction) > 10 and len(output.response) > 20:
            score += 0.3
        else:
            return 0.0
        chunk_sample = chunk_text[:200]
        def get_char_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        chunk_ngrams = get_char_ngrams(chunk_sample, 3)
        response_ngrams = get_char_ngrams(output.response, 3)
        if chunk_ngrams:
            overlap = len(chunk_ngrams & response_ngrams) / len(chunk_ngrams)
            score += min(overlap * 0.4, 0.4)
        else:
            common_chars = sum(1 for char in chunk_sample[:50] if char in output.response)
            score += min(common_chars / 50 * 0.4, 0.4)
        if len(output.response) > 50:
            score += 0.2
        if len(output.response) > 100:
            score += 0.1
        
        return min(score, 1.0)
    
    def _merge_outputs(self, outputs: List[ExpertOutput]) -> ExpertOutput:
        """Merge multiple expert outputs (e.g. pick best by quality_score)."""
        return max(outputs, key=lambda x: x.quality_score)

