"""
01 Text chunker (module 03).

Split long text into fixed-size chunks with overlap.
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Text chunker; supports paragraph and fixed-length splitting."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_length: int = 100):
        """
        Args:
            chunk_size: Target chunk size (characters)
            chunk_overlap: Overlap between chunks (characters)
            min_chunk_length: Minimum chunk length; shorter chunks are dropped
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs. Returns list of paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def chunk_text(self, text: str, doc_id: str = "") -> List[Dict[str, str]]:
        """Split text into chunks. Returns list of dicts with doc_id and text."""
        chunks: List[Dict[str, str]] = []

        paragraphs = self.split_by_paragraphs(text)

        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n{paragraph}" if current_chunk else paragraph
                continue

            if current_chunk and len(current_chunk) >= self.min_chunk_length:
                chunk_id = f"{doc_id}_chunk_{chunk_index}" if doc_id else f"chunk_{chunk_index}"
                chunks.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": current_chunk})
                chunk_index += 1

            if len(paragraph) > self.chunk_size:
                sentences = re.split(r"[。！？\n]", paragraph)
                current_chunk = ""

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                        current_chunk = f"{current_chunk}{sentence}" if current_chunk else sentence
                        continue

                    if current_chunk and len(current_chunk) >= self.min_chunk_length:
                        chunk_id = f"{doc_id}_chunk_{chunk_index}" if doc_id else f"chunk_{chunk_index}"
                        chunks.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": current_chunk})
                        chunk_index += 1

                    if self.chunk_overlap > 0 and current_chunk:
                        overlap_text = current_chunk[-self.chunk_overlap :]
                        current_chunk = overlap_text + sentence
                    else:
                        current_chunk = sentence
            else:
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + "\n" + paragraph
                else:
                    current_chunk = paragraph

        if current_chunk and len(current_chunk) >= self.min_chunk_length:
            chunk_id = f"{doc_id}_chunk_{chunk_index}" if doc_id else f"chunk_{chunk_index}"
            chunks.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": current_chunk})

        return chunks


