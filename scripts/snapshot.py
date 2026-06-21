#!/usr/bin/env python3
"""集成快照工具 — 功能集成完成后备份主入口 app.py 到 backups/snapshots/

用法：
    cd personal_assistant
    uv run python scripts/snapshot.py stage62_agent_graph "P1 /chat/graph 接入"

目录结构：
    backups/snapshots/20260621_stage62_agent_graph/
        MANIFEST.md          # 本次快照说明
        app.py               # 当时的入口文件
    backups/manifest.json    # 全部快照索引
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# 项目根目录（personal_assistant/）
ROOT = Path(__file__).resolve().parent.parent
BACKUPS_DIR = ROOT / "backups"
SNAPSHOTS_DIR = BACKUPS_DIR / "snapshots"
MANIFEST_FILE = BACKUPS_DIR / "manifest.json"

# 每次集成只备份主入口（类似 step23_rag.py → app.py 的里程碑文件）
ENTRY_FILES: list[str] = [
    "app.py",
]


def load_manifest() -> list[dict]:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return []


def save_manifest(entries: list[dict]) -> None:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def create_snapshot(label: str, note: str, files: list[str]) -> Path:
    date_prefix = datetime.now().strftime("%Y%m%d")
    snapshot_name = f"{date_prefix}_{label}"
    dest = SNAPSHOTS_DIR / snapshot_name

    if dest.exists():
        raise SystemExit(f"快照已存在，请换 label：{dest}")

    dest.mkdir(parents=True)

    copied: list[str] = []
    missing: list[str] = []

    for rel in files:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)

    manifest_md = dest / "MANIFEST.md"
    manifest_md.write_text(
        "\n".join([
            f"# 快照：{snapshot_name}",
            "",
            f"- **日期**：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- **阶段标签**：{label}",
            f"- **说明**：{note or '（无）'}",
            "",
            "## 包含文件",
            "",
            *[f"- `{f}`" for f in copied],
            "",
            *(["## 缺失文件", ""] + [f"- `{f}`（未找到，跳过）" for f in missing] if missing else []),
            "",
            "## 恢复方式",
            "",
            "```bash",
            "cp backups/snapshots/"
            + snapshot_name
            + "/app.py app.py",
            "```",
            "",
        ]),
        encoding="utf-8",
    )

    entries = load_manifest()
    entries.append({
        "name": snapshot_name,
        "label": label,
        "date": date_prefix,
        "note": note,
        "files": copied,
        "missing": missing,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_manifest(entries)

    print(f"✅ 快照已创建：{dest}")
    print(f"   文件数：{len(copied)}")
    if missing:
        print(f"   ⚠️ 缺失：{', '.join(missing)}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 app.py 集成快照到 backups/snapshots/")
    parser.add_argument("label", help="阶段标签，如 stage62_agent_graph")
    parser.add_argument("note", nargs="?", default="", help="本次集成说明")
    args = parser.parse_args()

    create_snapshot(args.label, args.note, list(ENTRY_FILES))


if __name__ == "__main__":
    main()
