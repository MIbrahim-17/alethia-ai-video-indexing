"""Copy this project's Claude Code chat history into ai_chats/ (brief §7).

The files are copied unchanged, except that the value of every secret in .env is replaced
with [REMOVED], as the brief allows. Fails if anything that looks like a token is left.
Run it again as the last step before the final push, because the current session keeps growing.

Usage: python3 tools/export_chats.py
"""

import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = Path.home() / ".claude" / "projects" / "-home-popchacha-Desktop-Alethia-AI"
DST = REPO / "ai_chats" / "claude_code"
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}")


def secrets():
    out = []
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if len(v) >= 6:
                out.append(v)
    return out


def main():
    vals = secrets()
    if DST.exists():
        shutil.rmtree(DST)
    n_files = n_removed = 0
    for src in sorted(SRC.rglob("*")):
        rel = src.relative_to(SRC)
        if not src.is_file() or rel.parts[0] == "memory":
            continue
        text = src.read_text(errors="surrogateescape")
        for v in vals:
            n_removed += text.count(v)
            text = text.replace(v, "[REMOVED]")
        left = TOKEN_RE.findall(text)
        if left:
            sys.exit(f"token-like string still in {rel}; not exported")
        out = DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, errors="surrogateescape")
        n_files += 1
    print(f"exported {n_files} files to {DST.relative_to(REPO)}; secrets replaced: {n_removed}")


if __name__ == "__main__":
    main()
