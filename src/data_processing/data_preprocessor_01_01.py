"""
01 Data preprocessing (module 01).

Parse, clean, and chunk from raw_corpus; output final chunks (recommended: chunks_clean.jsonl).
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import multiprocessing
import threading

from src.data_processing.document_parser_01_02 import DocumentParser
from src.data_processing.text_chunker_01_03 import TextChunker

try:
    from simhash import Simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False
    logging.warning("simhash not available, chunk-level deduplication will be disabled")

logger = logging.getLogger(__name__)


def _process_file_worker(args):
    """
    Worker to process a single file for multiprocessing.
    Must be defined at module level so multiprocessing can pickle it.
    """
    file_path_str, raw_corpus_dir_str, config_dict = args

    file_path = Path(file_path_str)
    raw_corpus_dir = Path(raw_corpus_dir_str)

    # Create a new preprocessor instance per process (avoid shared state)
    from src.data_processing.data_preprocessor_01_01 import DataPreprocessor

    worker_preprocessor = DataPreprocessor(
        config_path=config_dict.get("config_path"), project_root=Path(config_dict.get("project_root"))
    )

    try:
        chunks = worker_preprocessor.process_single_file(file_path, raw_corpus_dir, enable_audit=True)

        if not chunks:
            return {
                "file_path": str(file_path),
                "success": False,
                "skipped": True,
                "chunks": [],
                "audit_log": [],
            }

        return {
            "file_path": str(file_path),
            "success": True,
            "skipped": False,
            "chunks": chunks,
            "audit_log": worker_preprocessor.audit_log,
        }
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
        return {
            "file_path": str(file_path),
            "success": False,
            "skipped": False,
            "failed": True,
            "error": str(e),
            "chunks": [],
            "audit_log": [],
        }


class DataPreprocessor:
    """Data preprocessor."""

    def __init__(self, config_path: str = "configs/config.yaml", project_root: Optional[Path] = None):
        # Resolve project root
        if project_root is None:
            current = Path.cwd()
            if current.name == "notebooks":
                current = current.parent
            while current != current.parent:
                if (current / "configs").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()
        self.project_root = Path(project_root)

        # Load config (relative to project root)
        config_path_obj = Path(config_path)
        if config_path_obj.is_absolute():
            config_full_path = config_path_obj
        else:
            # Resolve relative path from project root (no resolve() to avoid cwd dependency)
            config_full_path = self.project_root / config_path

        self.config = self.load_config(str(config_full_path))

        self.parser = DocumentParser(
            convert_traditional=self.config["data"].get("convert_traditional", True),
            chinese_only=self.config["data"].get("chinese_only", True),
            min_chinese_ratio=self.config["data"].get("min_chinese_ratio", 0.60),
        )
        self.chunker = TextChunker(
            chunk_size=self.config["data"].get("chunk_size", 512),
            chunk_overlap=self.config["data"].get("chunk_overlap", 50),
            min_chunk_length=self.config["data"].get("min_chunk_length", 100),
        )
        
        # Cross-document chunk-level deduplication (simhash)
        self.chunk_simhashes: List[int] = []
        self.chunk_simhash_threshold = self.config["data"].get("chunk_simhash_threshold", 3)
        
        # Audit log
        self.audit_log: List[Dict] = []

    @staticmethod
    def load_config(config_path: str = "configs/config.yaml") -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_path(self, path: str) -> Path:
        path_obj = Path(path)
        return path_obj if path_obj.is_absolute() else (self.project_root / path)

    def get_all_files(self, directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
        if extensions is None:
            extensions = [".pdf", ".docx", ".txt"]
        files: List[Path] = []
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
        return sorted(files)

    def generate_doc_id(self, file_path: Path, raw_corpus_path: Path) -> str:
        relative_path = file_path.relative_to(raw_corpus_path)
        doc_id = str(relative_path).replace(os.sep, "_").replace("/", "_").replace("\\", "_")
        return os.path.splitext(doc_id)[0]

    def _is_chunk_duplicate(self, chunk_text: str) -> bool:
        """Check if chunk duplicates an existing one (simhash)."""
        if not SIMHASH_AVAILABLE or len(chunk_text) < 20:
            return False
        try:
            chunk_hash = Simhash(chunk_text).value
            for seen_hash in self.chunk_simhashes:
                if bin(chunk_hash ^ seen_hash).count('1') <= self.chunk_simhash_threshold:
                    return True
            self.chunk_simhashes.append(chunk_hash)
            return False
        except Exception as e:
            logger.warning(f"SimHash calculation failed for chunk: {e}")
            return False

    def process_single_file(
        self, 
        file_path: Path, 
        raw_corpus_path: Path,
        enable_audit: bool = True
    ) -> Optional[List[Dict]]:
        """
        Process a single file and return cleaned chunks.

        Args:
            file_path: Path to the file
            raw_corpus_path: Root path of raw corpus
            enable_audit: Whether to record audit log

        Returns:
            List of processed chunks, or None on failure
        """
        try:
            doc_id = self.generate_doc_id(file_path, raw_corpus_path)
            
            # Parse and clean (returns text and stats)
            text, parse_stats = self.parser.parse_and_clean(
                str(file_path),
                enable_line_dedup=True,
                enable_para_dedup=True,
                remove_structural_noise=True
            )
            
            min_len = self.config["data"].get("min_chunk_length", 100)
            original_text_len = len(text) if text else 0
            
            # Document-level quality check
            if not text or len(text.strip()) < min_len:
                if enable_audit:
                    self.audit_log.append({
                        "doc_id": doc_id,
                        "file_path": str(file_path),
                        "action": "rejected",
                        "reason": "Text too short or empty",
                        "original_length": original_text_len,
                        "final_length": 0,
                        "parse_stats": parse_stats
                    })
                return None

            ok, reason = self.parser.is_valid_text(text, min_len=min_len)
            if not ok:
                if enable_audit:
                    self.audit_log.append({
                        "doc_id": doc_id,
                        "file_path": str(file_path),
                        "action": "rejected",
                        "reason": reason,
                        "original_length": original_text_len,
                        "final_length": len(text),
                        "parse_stats": parse_stats
                    })
                return None

            # Chunk
            chunks = self.chunker.chunk_text(text, doc_id=doc_id)
            
            # Chunk-level cleaning + dedup
            cleaned_chunks: List[Dict] = []
            chunk_dedup_count = 0
            
            for ch in chunks:
                t = self.parser.sanitize_text(ch.get("text", ""))
                ok2, reason2 = self.parser.is_valid_text(t, min_len=min_len)
                if not ok2:
                    if enable_audit:
                        self.audit_log.append({
                            "doc_id": doc_id,
                            "chunk_id": ch.get("chunk_id", ""),
                            "action": "chunk_rejected",
                            "reason": reason2,
                            "original_length": len(ch.get("text", "")),
                            "final_length": 0
                        })
                    continue
                
                # Chunk-level simhash dedup (cross-document)
                if self._is_chunk_duplicate(t):
                    chunk_dedup_count += 1
                    if enable_audit:
                        self.audit_log.append({
                            "doc_id": doc_id,
                            "chunk_id": ch.get("chunk_id", ""),
                            "action": "chunk_deduplicated",
                            "reason": "Similar to existing chunk (simhash)",
                            "original_length": len(t),
                            "final_length": 0
                        })
                    continue
                
                ch["text"] = t
                cleaned_chunks.append(ch)
            
            # Record audit log for successful processing
            if enable_audit and cleaned_chunks:
                self.audit_log.append({
                    "doc_id": doc_id,
                    "file_path": str(file_path),
                    "action": "processed",
                    "original_length": original_text_len,
                    "final_length": len(text),
                    "chunks_count": len(cleaned_chunks),
                    "chunks_dedup_count": chunk_dedup_count,
                    "parse_stats": parse_stats
                })
            
            return cleaned_chunks if cleaned_chunks else None
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            if enable_audit:
                self.audit_log.append({
                    "doc_id": self.generate_doc_id(file_path, raw_corpus_path) if 'doc_id' not in locals() else doc_id,
                    "file_path": str(file_path),
                    "action": "error",
                    "reason": str(e)
                })
            return None

    def process_all_files_clean(
        self,
        raw_corpus_path: Optional[str] = None,
        output_path: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        show_progress: bool = True,
        enable_audit: bool = True,
    ) -> Dict:
        """
        Process all files; produce chunks_clean.jsonl and optional audit log.

        Args:
            raw_corpus_path: Raw corpus path
            output_path: Output chunks file path
            audit_log_path: Audit log path (None to skip)
            show_progress: Whether to show progress bar
            enable_audit: Whether to record audit log (in memory)

        Returns:
            Stats dictionary
        """
        if raw_corpus_path is None:
            raw_corpus_path = self.config["data"]["raw_corpus_path"]
        if output_path is None:
            processed_corpus_path = self.config["data"]["processed_corpus_path"]
            output_path = os.path.join(processed_corpus_path, "chunks_clean.jsonl")
        if audit_log_path is None and enable_audit:
            processed_corpus_path = self.config["data"]["processed_corpus_path"]
            audit_log_path = os.path.join(processed_corpus_path, "audit_log.jsonl")

        raw_corpus_dir = self._resolve_path(raw_corpus_path)
        output_file = self._resolve_path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Reset audit log and simhash set
        if enable_audit:
            self.audit_log = []
        self.chunk_simhashes = []

        all_files = self.get_all_files(raw_corpus_dir)
        stats = {
            "total_files": len(all_files),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_chunks": 0,
            "chunks_dedup_count": 0,
        }

        iterator = tqdm(all_files, desc="Processing files (clean)") if show_progress else all_files
        with open(output_file, "w", encoding="utf-8") as f:
            for file_path in iterator:
                chunks = self.process_single_file(file_path, raw_corpus_dir, enable_audit=enable_audit)
                if not chunks:
                    stats["skipped"] += 1
                    continue
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    stats["total_chunks"] += 1
                stats["successful"] += 1

        # Count chunk deduplication
        if enable_audit:
            dedup_count = sum(
                1 for log in self.audit_log if log.get("action") == "chunk_deduplicated"
            )
            stats["chunks_dedup_count"] = dedup_count
            
            # Write audit log
            if audit_log_path:
                audit_file = self._resolve_path(audit_log_path)
                audit_file.parent.mkdir(parents=True, exist_ok=True)
                with open(audit_file, "w", encoding="utf-8") as f_audit:
                    for log_entry in self.audit_log:
                        f_audit.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                stats["audit_log_path"] = str(audit_file)

        return stats

    def verify_processing(self, raw_corpus_path: Optional[str] = None, chunks_file: Optional[str] = None) -> Dict:
        if raw_corpus_path is None:
            raw_corpus_path = self.config["data"]["raw_corpus_path"]
        if chunks_file is None:
            processed_corpus_path = self.config["data"]["processed_corpus_path"]
            chunks_file = os.path.join(processed_corpus_path, "chunks_clean.jsonl")

        raw_corpus_dir = self._resolve_path(raw_corpus_path)
        chunks_path = self._resolve_path(chunks_file)

        all_files = self.get_all_files(raw_corpus_dir)
        raw_stats: Dict[str, Any] = {"total": len(all_files), "by_type": {}, "by_layer": {}}
        for file_path in all_files:
            ext = file_path.suffix.lower()
            raw_stats["by_type"][ext] = raw_stats["by_type"].get(ext, 0) + 1  # type: ignore
            parts = file_path.parts
            for layer in ["Layer1", "Layer2", "Layer3", "Layer4"]:
                if layer in parts:
                    raw_stats["by_layer"][layer] = raw_stats["by_layer"].get(layer, 0) + 1  # type: ignore
                    break

        processed_stats: Dict[str, Any] = {"total_chunks": 0, "unique_docs": set(), "empty_chunks": 0, "short_chunks": 0}
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    processed_stats["total_chunks"] += 1  # type: ignore
                    doc_id = chunk.get("doc_id", "")
                    if doc_id:
                        processed_stats["unique_docs"].add(doc_id)  # type: ignore
                    text = chunk.get("text", "")
                    if not text or len(text.strip()) == 0:
                        processed_stats["empty_chunks"] += 1  # type: ignore
                    elif len(text.strip()) < 50:
                        processed_stats["short_chunks"] += 1  # type: ignore

        processed_stats["unique_docs"] = len(processed_stats["unique_docs"])  # type: ignore
        coverage = (processed_stats["unique_docs"] / raw_stats["total"] * 100) if raw_stats["total"] > 0 else 0  # type: ignore

        return {"raw_stats": raw_stats, "processed_stats": processed_stats, "coverage": coverage, "chunks_path": str(chunks_path)}

    def process_all_files_with_checkpoint(
        self,
        raw_corpus_path: Optional[str] = None,
        output_path: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        show_progress: bool = True,
        enable_audit: bool = True,
        num_workers: Optional[int] = None,
        batch_size: int = 50,
        checkpoint_interval: int = 5,
    ) -> Dict:
        """
        Process all files with checkpoint (resumable after interrupt).

        Args:
            raw_corpus_path: Raw corpus path
            output_path: Output chunks file path
            audit_log_path: Audit log path (None to skip)
            checkpoint_path: Checkpoint path (None for default)
            show_progress: Whether to show progress bar
            enable_audit: Whether to record audit log (in memory)
            num_workers: Number of worker processes (None = CPU count, 1 = single process)
            batch_size: Files per batch
            checkpoint_interval: Save checkpoint every N batches

        Returns:
            Stats dictionary
        """
        if raw_corpus_path is None:
            raw_corpus_path = self.config["data"]["raw_corpus_path"]
        if output_path is None:
            processed_corpus_path = self.config["data"]["processed_corpus_path"]
            output_path = os.path.join(processed_corpus_path, "chunks_clean.jsonl")
        if audit_log_path is None and enable_audit:
            processed_corpus_path = self.config["data"]["processed_corpus_path"]
            audit_log_path = os.path.join(processed_corpus_path, "audit_log.jsonl")
        if checkpoint_path is None:
            processed_corpus_path = self.config["data"]["processed_corpus_path"]
            checkpoint_path = os.path.join(processed_corpus_path, "checkpoint.json")

        raw_corpus_dir = self._resolve_path(raw_corpus_path)
        output_file = self._resolve_path(output_path)
        checkpoint_file = self._resolve_path(checkpoint_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 1. Check for existing checkpoint
        checkpoint_data = None
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                logger.info("=" * 60)
                logger.info("Checkpoint found; resuming from last run")
                logger.info("=" * 60)
                logger.info(f"Checkpoint time: {checkpoint_data.get('timestamp', 'N/A')}")
                logger.info(f"Processed files: {checkpoint_data.get('processed_count', 0)}/{checkpoint_data.get('total_files', 0)}")
                logger.info(f"Chunks generated: {checkpoint_data.get('total_chunks', 0):,}")
                logger.info(f"Simhashes saved: {len(checkpoint_data.get('chunk_simhashes', [])):,}")
            except Exception as e:
                logger.warning(f"Checkpoint file corrupted: {e}; starting from scratch")
                checkpoint_data = None

        # 2. Get all files
        all_files = self.get_all_files(raw_corpus_dir)

        # 3. Initialize state (from checkpoint or new)
        if checkpoint_data:
            # Restore state
            processed_files = set(checkpoint_data.get("processed_files", []))
            self.chunk_simhashes = checkpoint_data.get("chunk_simhashes", [])
            if checkpoint_data.get("audit_log"):
                self.audit_log = checkpoint_data["audit_log"]
            else:
                self.audit_log = []

            # Find remaining files
            remaining_files = [f for f in all_files if str(f) not in processed_files]
            start_index = len(processed_files)

            # Open output file in append mode
            output_mode = "a"
            audit_mode = "a"

            logger.info(f"Resuming from file {start_index + 1} ({len(remaining_files)} files remaining)")
        else:
            # New state
            processed_files = set()
            self.chunk_simhashes = []
            self.audit_log = []
            remaining_files = all_files
            start_index = 0

            # Open output file in write mode (overwrite)
            output_mode = "w"
            audit_mode = "w"

        # 4. Initialize stats
        stats = {
            "total_files": len(all_files),
            "successful": checkpoint_data.get("stats", {}).get("successful", start_index) if checkpoint_data else start_index,
            "failed": checkpoint_data.get("stats", {}).get("failed", 0) if checkpoint_data else 0,
            "skipped": checkpoint_data.get("stats", {}).get("skipped", 0) if checkpoint_data else 0,
            "total_chunks": checkpoint_data.get("stats", {}).get("total_chunks", 0) if checkpoint_data else 0,
            "chunks_dedup_count": checkpoint_data.get("stats", {}).get("chunks_dedup_count", 0) if checkpoint_data else 0,
        }

        # 5. Decide single vs multiprocess
        use_multiprocessing = num_workers is None or num_workers > 1
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        if use_multiprocessing and len(remaining_files) > 0:
            # Use multiprocessing
            return self._process_with_multiprocessing(
                remaining_files=remaining_files,
                raw_corpus_dir=raw_corpus_dir,
                output_file=output_file,
                audit_log_path=audit_log_path,
                checkpoint_file=checkpoint_file,
                processed_files=processed_files,
                stats=stats,
                start_index=start_index,
                output_mode=output_mode,
                audit_mode=audit_mode,
                show_progress=show_progress,
                enable_audit=enable_audit,
                num_workers=num_workers,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
            )
        else:
            # Single process (simpler, slower)
            return self._process_single_process(
                remaining_files=remaining_files,
                raw_corpus_dir=raw_corpus_dir,
                output_file=output_file,
                audit_log_path=audit_log_path,
                checkpoint_file=checkpoint_file,
                processed_files=processed_files,
                stats=stats,
                start_index=start_index,
                output_mode=output_mode,
                audit_mode=audit_mode,
                show_progress=show_progress,
                enable_audit=enable_audit,
                batch_size=batch_size,
                checkpoint_interval=checkpoint_interval,
            )

    def _process_single_process(
        self,
        remaining_files: List[Path],
        raw_corpus_dir: Path,
        output_file: Path,
        audit_log_path: Optional[str],
        checkpoint_file: Path,
        processed_files: Set[str],
        stats: Dict,
        start_index: int,
        output_mode: str,
        audit_mode: str,
        show_progress: bool,
        enable_audit: bool,
        batch_size: int,
        checkpoint_interval: int,
    ) -> Dict:
        """Single-process with checkpoint."""
        iterator = tqdm(remaining_files, desc="Processing (checkpoint)", initial=start_index) if show_progress else remaining_files

        with open(output_file, output_mode, encoding="utf-8") as f_out:
            batch_count = 0
            for idx, file_path in enumerate(iterator):
                chunks = self.process_single_file(file_path, raw_corpus_dir, enable_audit=enable_audit)

                if not chunks:
                    stats["skipped"] += 1
                    processed_files.add(str(file_path))
                else:
                    # Write chunks (dedup in main process)
                    for chunk in chunks:
                        chunk_text = chunk.get("text", "")
                        if chunk_text and len(chunk_text) >= 20:
                            try:
                                if SIMHASH_AVAILABLE:
                                    chunk_hash = Simhash(chunk_text).value
                                    is_dup = False
                                    for seen_hash in self.chunk_simhashes:
                                        if bin(chunk_hash ^ seen_hash).count("1") <= self.chunk_simhash_threshold:
                                            is_dup = True
                                            stats["chunks_dedup_count"] = stats.get("chunks_dedup_count", 0) + 1
                                            break
                                    if not is_dup:
                                        self.chunk_simhashes.append(chunk_hash)
                                        f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                        stats["total_chunks"] += 1
                                else:
                                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                    stats["total_chunks"] += 1
                            except Exception:
                                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                stats["total_chunks"] += 1
                        else:
                            f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            stats["total_chunks"] += 1

                    stats["successful"] += 1
                    processed_files.add(str(file_path))

                # Periodic checkpoint save
                batch_count += 1
                if batch_count >= batch_size:
                    batch_count = 0
                    self._save_checkpoint(
                        checkpoint_file,
                        processed_files,
                        stats,
                        len(processed_files),
                        len(processed_files) + len(remaining_files),
                    )
                    if show_progress and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix({"checkpoint": f"{len(processed_files)}/{len(processed_files) + len(remaining_files)}"})

        # Final checkpoint save
        self._save_checkpoint(
            checkpoint_file,
            processed_files,
            stats,
            len(processed_files),
            len(processed_files) + len(remaining_files),
        )

        # Write audit log
        if enable_audit and audit_log_path:
            audit_file = self._resolve_path(audit_log_path)
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_file, audit_mode, encoding="utf-8") as f_audit:
                for log_entry in self.audit_log:
                    f_audit.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            stats["audit_log_path"] = str(audit_file)

        # Remove checkpoint file when done
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("\nDone; checkpoint file removed")

        return stats

    def _process_with_multiprocessing(
        self,
        remaining_files: List[Path],
        raw_corpus_dir: Path,
        output_file: Path,
        audit_log_path: Optional[str],
        checkpoint_file: Path,
        processed_files: Set[str],
        stats: Dict,
        start_index: int,
        output_mode: str,
        audit_mode: str,
        show_progress: bool,
        enable_audit: bool,
        num_workers: int,
        batch_size: int,
        checkpoint_interval: int,
    ) -> Dict:
        """Multi-process with checkpoint."""
        from multiprocessing import Pool

        # Split files into batches
        batches = [remaining_files[i : i + batch_size] for i in range(0, len(remaining_files), batch_size)]
        total_batches = len(batches)

        logger.info(f"Processing {len(remaining_files)} files in {total_batches} batches with {num_workers} workers")

        # Config for workers
        config_dict = {
            "config_path": str(self.project_root / "configs" / "config.yaml"),
            "project_root": str(self.project_root),
        }

        # Build arg list
        process_args = [(str(f), str(raw_corpus_dir), config_dict) for f in remaining_files]

        # Process using multiprocessing pool
        write_lock = threading.Lock()
        batch_results = []
        processed_count = start_index

        def write_batch_results(batch_results, f_out, stats, processed_files, chunk_simhashes, chunk_simhash_threshold):
            """Write batch results to file."""
            for res in batch_results:
                file_path_str = res["file_path"]

                if res.get("skipped"):
                    stats["skipped"] += 1
                    processed_files.add(file_path_str)
                elif res.get("failed"):
                    stats["failed"] += 1
                    processed_files.add(file_path_str)
                elif res.get("success"):
                    # Write chunks (dedup in main process)
                    for chunk in res["chunks"]:
                        chunk_text = chunk.get("text", "")
                        if chunk_text and len(chunk_text) >= 20:
                            try:
                                if SIMHASH_AVAILABLE:
                                    chunk_hash = Simhash(chunk_text).value
                                    is_dup = False
                                    # If simhashes list is large, only check recent entries
                                    max_check_count = 50000
                                    check_list = chunk_simhashes[-max_check_count:] if len(chunk_simhashes) > max_check_count else chunk_simhashes
                                    
                                    for seen_hash in check_list:
                                        if bin(chunk_hash ^ seen_hash).count("1") <= chunk_simhash_threshold:
                                            is_dup = True
                                            stats["chunks_dedup_count"] = stats.get("chunks_dedup_count", 0) + 1
                                            break
                                    if not is_dup:
                                        chunk_simhashes.append(chunk_hash)
                                        f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                        stats["total_chunks"] += 1
                                else:
                                    f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                    stats["total_chunks"] += 1
                            except Exception:
                                f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                                stats["total_chunks"] += 1
                        else:
                            f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            stats["total_chunks"] += 1

                    stats["successful"] += 1
                    processed_files.add(file_path_str)

                    if res.get("audit_log"):
                        # audit_log handled in main process
                        pass

        try:
            with Pool(processes=num_workers) as pool:
                # imap_unordered for efficiency
                results_iterator = pool.imap_unordered(_process_file_worker, process_args, chunksize=10)

                progress_bar = tqdm(
                    total=len(remaining_files), desc=f"Processing ({num_workers} workers, checkpoint)", initial=start_index
                )

                last_log_time = None
                for result in results_iterator:
                    batch_results.append(result)
                    processed_count += 1

                    # Each batch: write results and save checkpoint
                    if len(batch_results) >= batch_size or processed_count == len(remaining_files) + start_index:
                        import time
                        batch_start_time = time.time()
                        
                        # Write chunks and update state (with lock)
                        with write_lock:
                            write_start_time = time.time()
                            with open(output_file, "a", encoding="utf-8") as f_out:
                                write_batch_results(
                                    batch_results,
                                    f_out,
                                    stats,
                                    processed_files,
                                    self.chunk_simhashes,
                                    self.chunk_simhash_threshold,
                                )
                                for res in batch_results:
                                    if res.get("audit_log"):
                                        self.audit_log.extend(res["audit_log"])
                            write_time = time.time() - write_start_time
                            
                            if write_time > 10:
                                logger.warning(
                                    f"Batch write took {write_time:.1f}s, "
                                    f"simhashes: {len(self.chunk_simhashes):,}, "
                                    f"batch size: {len(batch_results)}"
                                )

                        batch_results = []
                        batch_time = time.time() - batch_start_time

                        # Periodic checkpoint and log
                        current_batch_num = (processed_count - start_index) // batch_size
                        if current_batch_num > 0 and current_batch_num % checkpoint_interval == 0:
                            checkpoint_start = time.time()
                            self._save_checkpoint(
                                checkpoint_file,
                                processed_files,
                                stats,
                                processed_count,
                                len(remaining_files) + start_index,
                            )
                            checkpoint_time = time.time() - checkpoint_start
                            progress_bar.set_postfix({
                                "checkpoint": f"{processed_count}/{len(remaining_files) + start_index}",
                                "simhashes": f"{len(self.chunk_simhashes):,}"
                            })
                            if checkpoint_time > 5:
                                logger.warning(f"Checkpoint save took {checkpoint_time:.1f}s")
                        
                        if current_batch_num > 0 and current_batch_num % 100 == 0:
                            logger.info(
                                f"Progress: {processed_count}/{len(remaining_files) + start_index} "
                                f"({processed_count/(len(remaining_files) + start_index)*100:.1f}%), "
                                f"simhashes: {len(self.chunk_simhashes):,}, "
                                f"chunks: {stats['total_chunks']:,}"
                            )

                    progress_bar.update(1)

                # Flush remaining batch
                if batch_results:
                    with write_lock:
                        with open(output_file, "a", encoding="utf-8") as f_out:
                            write_batch_results(
                                batch_results,
                                f_out,
                                stats,
                                processed_files,
                                self.chunk_simhashes,
                                self.chunk_simhash_threshold,
                            )
                            # Collect audit log
                            for res in batch_results:
                                if res.get("audit_log"):
                                    self.audit_log.extend(res["audit_log"])

                progress_bar.close()

        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user")
            logger.info("Checkpoint saved; will resume on next run")
            self._save_checkpoint(checkpoint_file, processed_files, stats, processed_count, len(remaining_files) + start_index)
            raise
        except Exception as e:
            logger.error(f"\nError during processing: {e}")
            logger.info("Checkpoint saved; fix and re-run to resume")
            self._save_checkpoint(checkpoint_file, processed_files, stats, processed_count, len(remaining_files) + start_index)
            raise

        # Write audit log
        if enable_audit and audit_log_path:
            audit_file = self._resolve_path(audit_log_path)
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_file, audit_mode, encoding="utf-8") as f_audit:
                for log_entry in self.audit_log:
                    f_audit.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            stats["audit_log_path"] = str(audit_file)

        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("\nDone; checkpoint file removed")

        return stats

    def _save_checkpoint(
        self, checkpoint_file: Path, processed_files: Set[str], stats: Dict, processed_count: int, total_files: int
    ):
        """Save checkpoint."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "total_files": total_files,
            "processed_count": processed_count,
            "processed_files": list(processed_files),
            "chunk_simhashes": self.chunk_simhashes,
            "audit_log": self.audit_log,
            "stats": stats.copy(),
        }
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f_cp:
            json.dump(checkpoint_data, f_cp, ensure_ascii=False, indent=2)


