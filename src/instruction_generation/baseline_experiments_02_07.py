"""
02-07 Baseline comparison experiments.

Self-Instruct, Alpaca, WizardLM, GPT-4 direct, human (upper bound).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaselineExperiment:
    """Baseline experiment runner."""
    def __init__(self, config: Dict):
        self.config = config
        self.api_client = None
        self._init_api_client()
    def _init_api_client(self):
        """Initialize API client."""
        api_provider = self.config.get("api_provider", "openai")
        if api_provider == "openai":
            try:
                import openai
                self.api_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.error("openai not installed")
    def self_instruct_generate(
        self,
        chunk_text: str,
        seed_instructions: Optional[List[str]] = None
    ) -> Dict:
        """Self-Instruct: generate instruction-answer pair from chunk."""
        if seed_instructions is None:
            seed_instructions = [
                "Summarize the main content of the following text.",
                "Answer the following question.",
                "Extract key information from the following text.",
                "Analyze ESG-related content in the following text."
            ]
        import random
        selected_seed = random.choice(seed_instructions)
        prompt = f"""Based on the text below, generate one instruction-answer pair.
Seed template: {selected_seed}

Text:
{chunk_text}

Output:
1. One concrete instruction (based on the seed, tailored to the text)
2. The corresponding answer

Format:
Instruction: [your instruction]
Answer: [your answer]"""
        try:
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            result_text = response.choices[0].message.content
            instruction, answer = self._parse_response(result_text)
            return {
                "instruction": instruction,
                "input": chunk_text,
                "output": answer,
                "method": "self_instruct",
                "seed_instruction": selected_seed
            }
        except Exception as e:
            logger.error(f"Self-Instruct failed: {e}")
            return None
    def alpaca_generate(
        self,
        chunk_text: str,
        template: Optional[str] = None
    ) -> Dict:
        """Alpaca-style generation with template."""
        if template is None:
            template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        instruction_prompt = f"""Based on the text below, generate one ESG-related instruction.

Text:
{chunk_text}

Generate one concrete instruction for an ESG task (e.g. summarize, Q&A, extract). Output only the instruction."""
        try:
            response1 = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": instruction_prompt}],
                temperature=0.7,
                max_tokens=512
            )
            instruction = response1.choices[0].message.content.strip()
            answer_prompt = template.format(instruction=instruction, input=chunk_text)
            response2 = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            answer = response2.choices[0].message.content.strip()
            return {
                "instruction": instruction,
                "input": chunk_text,
                "output": answer,
                "method": "alpaca",
                "template": template
            }
        except Exception as e:
            logger.error(f"Alpaca failed: {e}")
            return None
    def wizardlm_generate(
        self,
        chunk_text: str,
        evolution_rounds: int = 2
    ) -> Dict:
        """WizardLM-style evolution: refine instruction then generate answer."""
        initial_prompt = f"""Based on the text below, generate an initial ESG-related instruction.

Text:
{chunk_text}

Output one initial instruction only."""
        try:
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0.7,
                max_tokens=512
            )
            instruction = response.choices[0].message.content.strip()
            for round in range(evolution_rounds):
                evolution_prompt = f"""Improve the following instruction to be more specific, clear, and challenging.

Current instruction: {instruction}

Text:
{chunk_text}

Output the improved instruction only."""
                response = self.api_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": evolution_prompt}],
                    temperature=0.7,
                    max_tokens=512
                )
                instruction = response.choices[0].message.content.strip()
            answer_prompt = f"""Instruction: {instruction}

Input: {chunk_text}

Generate the response."""
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            answer = response.choices[0].message.content.strip()
            return {
                "instruction": instruction,
                "input": chunk_text,
                "output": answer,
                "method": "wizardlm",
                "evolution_rounds": evolution_rounds
            }
        except Exception as e:
            logger.error(f"WizardLM failed: {e}")
            return None
    def gpt4_direct_generate(self, chunk_text: str) -> Dict:
        """GPT-4 direct generation (no CoDE)."""
        prompt = f"""Based on the following ESG-related text, generate one high-quality instruction-answer pair.

Text:
{chunk_text}

Output:
1. One concrete ESG-related instruction
2. The corresponding answer

Format:
Instruction: [instruction]
Answer: [answer]"""
        try:
            response = self.api_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            result_text = response.choices[0].message.content
            instruction, answer = self._parse_response(result_text)
            return {
                "instruction": instruction,
                "input": chunk_text,
                "output": answer,
                "method": "gpt4_direct"
            }
        except Exception as e:
            logger.error(f"GPT-4 direct failed: {e}")
            return None
    def _parse_response(self, response_text: str) -> tuple:
        """Parse response to extract instruction and answer (supports Chinese/English labels)."""
        lines = response_text.strip().split('\n')
        instruction = ""
        answer = ""
        for line in lines:
            if line.startswith("指令：") or line.startswith("指令:") or line.lower().startswith("instruction:"):
                instruction = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("回答：") or line.startswith("回答:") or line.lower().startswith("answer:"):
                answer = line.split("：", 1)[-1].split(":", 1)[-1].strip()
        if not instruction:
            instruction = "Answer the following question."
        if not answer:
            answer = response_text.strip()
        return instruction, answer
    def run_baseline_comparison(
        self,
        chunks: List[Dict],
        methods: List[str] = None,
        output_dir: str = "results/baselines",
        sample_size: Optional[int] = None
    ) -> Dict:
        """Run baseline comparison; returns per-method stats."""
        if methods is None:
            methods = ["self_instruct", "alpaca", "wizardlm", "gpt4_direct"]
        if sample_size and sample_size > 0:
            import random
            if len(chunks) > sample_size:
                chunks = random.sample(chunks, sample_size)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stats = {}
        for method in methods:
            logger.info(f"Running baseline: {method}")
            results = []
            for chunk in tqdm(chunks, desc=f"Generate ({method})"):
                chunk_text = chunk.get("text", "")
                if not chunk_text:
                    continue
                try:
                    if method == "self_instruct":
                        result = self.self_instruct_generate(chunk_text)
                    elif method == "alpaca":
                        result = self.alpaca_generate(chunk_text)
                    elif method == "wizardlm":
                        result = self.wizardlm_generate(chunk_text)
                    elif method == "gpt4_direct":
                        result = self.gpt4_direct_generate(chunk_text)
                    else:
                        logger.warning(f"Unknown method: {method}")
                        continue
                    if result:
                        result["source_chunk_id"] = chunk.get("chunk_id", "")
                        result["source_doc_id"] = chunk.get("doc_id", "")
                        results.append(result)
                except Exception as e:
                    logger.error(f"Generation failed ({method}): {e}")
                    continue
            output_file = output_path / f"{method}_results.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            stats[method] = {
                "total": len(chunks),
                "successful": len(results),
                "output_file": str(output_file)
            }
            logger.info(f"{method}: {len(results)}/{len(chunks)} samples")
        return stats

