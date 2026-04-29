"""
03-02 Data Loader: Load and tokenize Alpaca-format instruction datasets.

Handles loading from JSONL, Alpaca prompt template formatting, tokenization,
train/eval splitting, and data statistics reporting.

Naming convention: filename_03_02.py (notebook 03, 2nd module)
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  ALPACA PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

# Template without input field (our primary case: input="" in train_filtered.jsonl)
ALPACA_TEMPLATE_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

# Template with input field (used for metadata version with chunk as input)
ALPACA_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

# Chinese-specific template (optional, may perform better for Chinese models)
ALPACA_TEMPLATE_ZH_NO_INPUT = (
    "以下是一个描述任务的指令。请编写一个适当完成请求的回复。\n\n"
    "### 指令：\n{instruction}\n\n"
    "### 回复：\n{output}"
)

ALPACA_TEMPLATE_ZH_WITH_INPUT = (
    "以下是一个描述任务的指令，并附有进一步的上下文信息。"
    "请编写一个适当完成请求的回复。\n\n"
    "### 指令：\n{instruction}\n\n"
    "### 输入：\n{input}\n\n"
    "### 回复：\n{output}"
)

TEMPLATE_MAP = {
    "alpaca_en": (ALPACA_TEMPLATE_NO_INPUT, ALPACA_TEMPLATE_WITH_INPUT),
    "alpaca_zh": (ALPACA_TEMPLATE_ZH_NO_INPUT, ALPACA_TEMPLATE_ZH_WITH_INPUT),
}


@dataclass
class DatasetStats:
    """Statistics for a loaded dataset."""
    total_records: int = 0
    avg_instruction_len: float = 0.0
    avg_output_len: float = 0.0
    max_instruction_len: int = 0
    max_output_len: int = 0
    avg_total_tokens: float = 0.0
    max_total_tokens: int = 0
    records_exceeding_max_len: int = 0
    template_name: str = ""

    def print_summary(self) -> None:
        """Print formatted statistics."""
        print(f"  Total records:          {self.total_records:>8,}")
        print(f"  Template:               {self.template_name}")
        print(f"  Avg instruction len:    {self.avg_instruction_len:>8.0f} chars")
        print(f"  Avg output len:         {self.avg_output_len:>8.0f} chars")
        print(f"  Max instruction len:    {self.max_instruction_len:>8,} chars")
        print(f"  Max output len:         {self.max_output_len:>8,} chars")
        if self.avg_total_tokens > 0:
            print(f"  Avg total tokens:       {self.avg_total_tokens:>8.0f}")
            print(f"  Max total tokens:       {self.max_total_tokens:>8,}")
            print(f"  Exceeding max_length:   {self.records_exceeding_max_len:>8,}")


# ═══════════════════════════════════════════════════════════════════
#  CORE DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_jsonl(file_path: str | Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dicts, each representing one record.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    records = []
    n_errors = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                n_errors += 1
                if n_errors <= 5:
                    logger.warning(f"JSON decode error at line {i}: {e}")

    logger.info(f"Loaded {len(records):,} records from {file_path.name} "
                f"({n_errors} errors)")
    return records


def validate_alpaca_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Validate that records conform to Alpaca format.

    Returns:
        (valid_records, invalid_records)
    """
    valid = []
    invalid = []
    for i, rec in enumerate(records):
        instruction = rec.get("instruction", "").strip()
        output = rec.get("output", "").strip()

        if not instruction:
            invalid.append({"index": i, "reason": "empty_instruction", "record": rec})
        elif not output:
            invalid.append({"index": i, "reason": "empty_output", "record": rec})
        else:
            valid.append(rec)

    if invalid:
        logger.warning(f"Found {len(invalid)} invalid records "
                       f"(empty instruction or output)")
    return valid, invalid


def format_alpaca_prompt(
    record: Dict[str, Any],
    template_name: str = "alpaca_en",
    include_output: bool = True,
) -> str:
    """Format a record into an Alpaca prompt string.

    Args:
        record: Dict with 'instruction', 'input' (optional), 'output'.
        template_name: Template variant ('alpaca_en' or 'alpaca_zh').
        include_output: Whether to include the output (True for training, False for inference).

    Returns:
        Formatted prompt string.
    """
    templates = TEMPLATE_MAP.get(template_name)
    if templates is None:
        raise ValueError(f"Unknown template: {template_name}. "
                         f"Choose from: {list(TEMPLATE_MAP.keys())}")

    no_input_tmpl, with_input_tmpl = templates
    instruction = record.get("instruction", "")
    inp = record.get("input", "").strip()
    output = record.get("output", "") if include_output else ""

    if inp:
        text = with_input_tmpl.format(
            instruction=instruction, input=inp, output=output
        )
    else:
        text = no_input_tmpl.format(instruction=instruction, output=output)

    return text


# ═══════════════════════════════════════════════════════════════════
#  DATASET PREPARATION FOR HUGGINGFACE TRAINER
# ═══════════════════════════════════════════════════════════════════

def prepare_tokenized_dataset(
    records: List[Dict],
    tokenizer: Any,
    max_length: int = 2048,
    template_name: str = "alpaca_en",
) -> Any:
    """Prepare a HuggingFace Dataset with tokenized Alpaca prompts.

    Args:
        records: List of Alpaca-format dicts.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        template_name: Prompt template variant.

    Returns:
        HuggingFace Dataset object ready for Trainer.
    """
    from datasets import Dataset

    texts = [
        format_alpaca_prompt(rec, template_name=template_name) for rec in records
    ]

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return dataset


def train_eval_split(
    records: List[Dict],
    eval_ratio: float = 0.05,
    seed: int = 42,
    stratify_by: Optional[str] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Split records into train and eval sets.

    Args:
        records: Full dataset records.
        eval_ratio: Fraction to hold out for evaluation.
        seed: Random seed.
        stratify_by: Optional metadata key for stratified splitting.

    Returns:
        (train_records, eval_records)
    """
    import random
    rng = random.Random(seed)

    if stratify_by:
        # Group by stratification key
        from collections import defaultdict
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for rec in records:
            key = str(rec.get("metadata", {}).get(stratify_by, "unknown"))
            groups[key].append(rec)

        train_recs, eval_recs = [], []
        for group_records in groups.values():
            rng.shuffle(group_records)
            n_eval = max(1, int(len(group_records) * eval_ratio))
            eval_recs.extend(group_records[:n_eval])
            train_recs.extend(group_records[n_eval:])
    else:
        shuffled = records.copy()
        rng.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * eval_ratio))
        eval_recs = shuffled[:n_eval]
        train_recs = shuffled[n_eval:]

    logger.info(f"Split: {len(train_recs):,} train, {len(eval_recs):,} eval "
                f"(ratio={eval_ratio})")
    return train_recs, eval_recs


def train_val_test_split(
    records: List[Dict],
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split records into train, validation, and test sets.

    Three-way split to avoid data leakage: validation is used during training
    for checkpoint selection; test is held out for final evaluation only.

    Args:
        records: Full dataset records.
        val_ratio: Fraction for validation (used during training).
        test_ratio: Fraction for test (held-out, final evaluation only).
        seed: Random seed for reproducibility.

    Returns:
        (train_records, val_records, test_records)
    """
    import random
    rng = random.Random(seed)

    shuffled = records.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_recs = shuffled[:n_test]
    val_recs = shuffled[n_test:n_test + n_val]
    train_recs = shuffled[n_test + n_val:]

    logger.info(
        f"3-way split (seed={seed}): "
        f"{len(train_recs):,} train, {len(val_recs):,} val, {len(test_recs):,} test "
        f"(ratios={1 - val_ratio - test_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})"
    )
    return train_recs, val_recs, test_recs


# ═══════════════════════════════════════════════════════════════════
#  DATA STATISTICS & REPORTING
# ═══════════════════════════════════════════════════════════════════

def compute_dataset_stats(
    records: List[Dict],
    tokenizer: Any = None,
    max_length: int = 2048,
    template_name: str = "alpaca_en",
) -> DatasetStats:
    """Compute comprehensive statistics for the dataset.

    Args:
        records: List of Alpaca-format dicts.
        tokenizer: Optional tokenizer for token-level stats.
        max_length: Max sequence length for truncation counting.
        template_name: Prompt template name.

    Returns:
        DatasetStats object.
    """
    stats = DatasetStats()
    stats.total_records = len(records)
    stats.template_name = template_name

    if not records:
        return stats

    inst_lens = [len(r.get("instruction", "")) for r in records]
    out_lens = [len(r.get("output", "")) for r in records]

    stats.avg_instruction_len = sum(inst_lens) / len(inst_lens)
    stats.avg_output_len = sum(out_lens) / len(out_lens)
    stats.max_instruction_len = max(inst_lens)
    stats.max_output_len = max(out_lens)

    if tokenizer is not None:
        token_counts = []
        for rec in records:
            text = format_alpaca_prompt(rec, template_name=template_name)
            tokens = tokenizer(text, truncation=False)["input_ids"]
            token_counts.append(len(tokens))

        stats.avg_total_tokens = sum(token_counts) / len(token_counts)
        stats.max_total_tokens = max(token_counts)
        stats.records_exceeding_max_len = sum(
            1 for tc in token_counts if tc > max_length
        )

    return stats


def compute_token_length_distribution(
    records: List[Dict],
    tokenizer: Any,
    template_name: str = "alpaca_en",
    bins: Optional[List[int]] = None,
) -> Dict[str, int]:
    """Compute token length distribution in bins.

    Args:
        records: Dataset records.
        tokenizer: HuggingFace tokenizer.
        template_name: Prompt template name.
        bins: Bin edges (default: [0, 256, 512, 1024, 1536, 2048, 4096, inf]).

    Returns:
        Dict mapping bin label to count.
    """
    if bins is None:
        bins = [0, 256, 512, 1024, 1536, 2048, 4096, float("inf")]

    token_counts = []
    for rec in records:
        text = format_alpaca_prompt(rec, template_name=template_name)
        tokens = tokenizer(text, truncation=False)["input_ids"]
        token_counts.append(len(tokens))

    distribution = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        label = f"[{lo}, {hi})" if hi != float("inf") else f"[{lo}, ∞)"
        count = sum(1 for tc in token_counts if lo <= tc < hi)
        distribution[label] = count

    return distribution
