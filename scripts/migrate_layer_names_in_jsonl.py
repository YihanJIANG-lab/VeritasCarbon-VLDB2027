#!/usr/bin/env python3
"""
Replace legacy layer names with English in JSONL files (doc_id, chunk_id).
Usage: python scripts/migrate_layer_names_in_jsonl.py [file1.jsonl ...]
If no files given, runs on data/processed_corpus/*.jsonl and data/instructions/*.jsonl.
"""
import json
import sys
from pathlib import Path

REPLACEMENTS = [
    ("第一层", "Layer1"),
    ("第二层", "Layer2"),
    ("第三层", "Layer3"),
    ("第四层", "Layer4"),
]

def migrate_line(line: str) -> str:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line
    changed = False
    for key in ("doc_id", "chunk_id"):
        if key in obj and isinstance(obj[key], str):
            new_val = obj[key]
            for cn, en in REPLACEMENTS:
                if cn in new_val:
                    new_val = new_val.replace(cn, en)
                    changed = True
            obj[key] = new_val
    return json.dumps(obj, ensure_ascii=False) + "\n" if changed else line

def migrate_file(path: Path, dry_run: bool = False) -> int:
    count = 0
    out_path = path.with_suffix(path.suffix + ".migrated") if not dry_run else None
    with open(path, "r", encoding="utf-8") as f_in:
        if not dry_run and out_path:
            f_out = open(out_path, "w", encoding="utf-8")
        try:
            for line in f_in:
                if not line.strip():
                    if not dry_run and out_path:
                        f_out.write(line)
                    continue
                new_line = migrate_line(line)
                if new_line != line:
                    count += 1
                if not dry_run and out_path:
                    f_out.write(new_line)
        finally:
            if not dry_run and out_path:
                f_out.close()
    if not dry_run and out_path and count > 0:
        out_path.replace(path)
    return count

def main():
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run" and not a.startswith("--")]
    project_root = Path(__file__).resolve().parent.parent
    if not args:
        files = list((project_root / "data" / "processed_corpus").glob("*.jsonl"))
        files += list((project_root / "data" / "instructions").glob("*.jsonl"))
    else:
        files = [Path(a) for a in args]
    total = 0
    for path in files:
        if not path.is_file():
            continue
        n = migrate_file(path, dry_run=dry_run)
        total += n
        print(f"{path}: {n} lines updated")
    print(f"Total lines updated: {total}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
