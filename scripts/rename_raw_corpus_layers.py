#!/usr/bin/env python3
"""Rename raw_corpus layer folders from Chinese to English: 第一层->Layer1, etc."""
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent
    raw = project_root / "data" / "raw_corpus"
    if not raw.is_dir():
        print("data/raw_corpus not found")
        return 1
    renames = [
        ("第一层", "Layer1"),
        ("第二层", "Layer2"),
        ("第三层", "Layer3"),
        ("第四层", "Layer4"),
    ]
    for old, new in renames:
        old_path = raw / old
        new_path = raw / new
        if old_path.is_dir():
            if new_path.exists():
                print(f"Skip {old}: {new} already exists")
            else:
                old_path.rename(new_path)
                print(f"Renamed: {old} -> {new}")
        else:
            print(f"Not found: {old_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
