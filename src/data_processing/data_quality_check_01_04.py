"""
01 Data quality check (module 04).

Quality check, clean, and filter JSONL files (e.g. chunks.jsonl, chunks_clean.jsonl) with tqdm.
"""

# pyright: reportMissingTypeStubs=false

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict
import logging

from tqdm import tqdm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Data quality checker."""

    def __init__(self):
        pass

    @staticmethod
    def _count_private_use(text: str) -> int:
        return len(re.findall(r"[\uE000-\uF8FF]", text))

    @staticmethod
    def _count_replacement(text: str) -> int:
        return text.count("\uFFFD")

    @staticmethod
    def _count_extb(text: str) -> int:
        return len(re.findall(r"[\U00020000-\U0002A6DF]", text))

    @staticmethod
    def sanitize_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\uE000-\uF8FF]", "", text)
        text = text.replace("\uFFFD", "")
        text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)
        text = re.sub(r"[ \u3000]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def is_valid_text(self, text: str) -> Tuple[bool, str]:
        if not text or len(text.strip()) < 10:
            return False, "Text too short or empty"

        total_chars = len(text)
        if total_chars == 0:
            return False, "Text is empty"

        if self._count_private_use(text) > 0:
            return False, "Contains private-use chars (possible encoding/font issue)"
        if self._count_replacement(text) > 0:
            return False, "Contains replacement char \\uFFFD (possible encoding issue)"
        extb = self._count_extb(text)
        if extb >= 6 or (extb / total_chars) > 0.01:
            return False, "Ext-B character ratio too high (possible garbled text)"

        common_chinese = len(re.findall(r"[\u4e00-\u9fa5]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))
        digits = len(re.findall(r"\d", text))
        common_punctuation = len(re.findall(r"[，。！？、；：\"\"''（）【】《》·—\s,.!?;:()\[\]<>/\\\-]", text))

        valid_chars = common_chinese + english_chars + digits + common_punctuation
        valid_ratio = valid_chars / total_chars
        if valid_ratio < 0.70:
            return False, f"Valid character ratio too low ({valid_ratio:.2%})"

        uncommon_chars = len(
            re.findall(r"[^\u0000-\u007F\u4e00-\u9fa5\s，。！？、；：\"\"''（）【】《》·—,.!?;:()\[\]<>/\\\-]", text)
        )
        uncommon_ratio = uncommon_chars / total_chars
        if uncommon_ratio > 0.15:
            return False, f"Uncommon character ratio too high ({uncommon_ratio:.2%})"

        chinese_ratio = common_chinese / total_chars
        english_ratio = english_chars / total_chars
        if chinese_ratio < 0.15 and english_ratio < 0.15:
            return False, "Neither predominantly Chinese nor English"

        return True, "Valid text"

    def filter_or_clean_chunks_with_progress(
        self,
        input_file: Path,
        output_file: Path,
        total_lines: Optional[int] = None,
        show_progress: bool = True,
        mode: str = "clean",
    ) -> Dict[str, Any]:
        """
        mode:
          - strict: fail then discard
          - clean: on fail, sanitize then re-check; write cleaned text if salvaged
        """
        if mode not in {"strict", "clean"}:
            raise ValueError("mode must be 'strict' or 'clean'")

        stats: Dict[str, Any] = {
            "total_chunks": 0,
            "valid_chunks": 0,
            "invalid_chunks": 0,
            "salvaged_chunks": 0,
            "invalid_reasons": defaultdict(int),
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f_in:
            iterator: Any = f_in
            if show_progress:
                iterator = tqdm(
                    f_in,
                    total=total_lines,
                    desc=f"{'Clean filter' if mode=='clean' else 'Strict filter'}",
                    unit="lines",
                )

            with open(output_file, "w", encoding="utf-8") as f_out:
                for line in iterator:
                    try:
                        chunk = json.loads(line.strip())
                        stats["total_chunks"] += 1
                        text = chunk.get("text", "")

                        ok, reason = self.is_valid_text(text)
                        if not ok and mode == "clean":
                            cleaned = self.sanitize_text(text)
                            if cleaned and cleaned != text:
                                ok2, _ = self.is_valid_text(cleaned)
                                if ok2:
                                    chunk["text"] = cleaned
                                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                    stats["valid_chunks"] += 1
                                    stats["salvaged_chunks"] += 1
                                    continue

                        if ok:
                            f_out.write(line)
                            stats["valid_chunks"] += 1
                        else:
                            stats["invalid_chunks"] += 1
                            stats["invalid_reasons"][reason] += 1
                    except json.JSONDecodeError:
                        stats["invalid_chunks"] += 1
                        stats["invalid_reasons"]["JSON parse error"] += 1
                        continue

        return stats


