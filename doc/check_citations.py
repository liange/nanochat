#!/usr/bin/env python3
"""
Citations checker for the Chinese technical docs in this directory.

Why this exists
---------------
The docs used to cite source positions as bare `file.py:123` line numbers. Upstream
refactors (deleting the report module and the WebUI, merging the two optimizer
implementations, adding the inference-cost helpers to gpt.py, ...) invalidated a large
fraction of them, and nothing caught it because a stale line number is still "valid"
text. This script makes the failure mode loud: it checks every citation in every
markdown file in this directory and fails if an anchor no longer resolves.

Citation syntax
---------------
    path/to/file.py#symbol        preferred: resolve `symbol` in that file
    path/to/file.py#symbol@L42    additionally assert line 42 is inside `symbol`
    path/to/file.py:123           legacy bare line number, range-checked only
    path/to/file.py:123 "text"    legacy line number + assert `text` is on that line

`#symbol` is durable across refactors (it only breaks if the symbol itself is renamed
or removed, which is real signal). A bare line number is not, so prefer `#symbol`.

Lines containing the marker `doc:no-check` (e.g. inside an HTML comment) are skipped
entirely, which is the escape hatch for deliberate counter-examples.

Usage
-----
    python doc/check_citations.py            # check, exit 1 on any error
    python doc/check_citations.py --verbose  # also list every resolved citation

Run it after editing any doc file. `pytest tests/ -k citations` runs it too.
"""

import argparse
import ast
import os
import re
import sys

# -----------------------------------------------------------------------------
# Citation parsing

# Anchor characters are deliberately narrow: anything that can legally follow an
# anchor in prose (#, ,, ., ), backtick, whitespace, ...) terminates it. A lazy
# quantifier here silently truncates `#GPT.estimate_flops` to `#GPT`, so this is greedy.
# Markdown headings may contain spaces (`#Precision / dtype`), so those anchors get a
# separate, wider pattern.
_ANCHOR = r"[A-Za-z0-9_][A-Za-z0-9_.\-]*"
_ANCHOR_MD = r"[A-Za-z0-9_][A-Za-z0-9_.\-/ ]*[A-Za-z0-9_.\-/]|[A-Za-z0-9_]"
_QUOTE = "[^\u0060\"\\n]{4,80}"

# The trailing `(?![\w.\-])` stops the extension match from cutting a longer name short
# (`.` is not a word char, so without it `p.shape` would match as a `.sh` script).
_PATH = r"(?<![\w/.-])(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sh|html|toml|lock))(?![\w.\-])"
_PATH_MD = r"(?<![\w/.-])(?P<path>[A-Za-z0-9_][A-Za-z0-9_./-]*\.md)(?![\w.\-])"
_TAIL_HEAD = (
    r"(?:"
    r"#(?P<anchor>"
)
_TAIL_BODY = (
    r")"
    r"(?:@L(?P<anchor_line>\d+))?"
    r"|:(?P<line>\d+)(?:-L?(?P<line_end>\d+))?"
    r")?"
    r"(?:\s+(?:\u0060(?P<quote1>" + _QUOTE + r")\u0060|\"(?P<quote2>" + _QUOTE + r")\"))?"
)

# `nanochat/gpt.py#GPT.estimate_flops` / `scripts/base_train.py:263`
CITATION_RE = re.compile(_PATH + _TAIL_HEAD + _ANCHOR + _TAIL_BODY)
# Markdown headings may contain spaces: `README.md#Precision / dtype`
CITATION_RE_MD = re.compile(_PATH_MD + _TAIL_HEAD + _ANCHOR_MD + _TAIL_BODY)

SKIP_MARKER = "doc:no-check"
SKIP_OPEN = "doc:no-check-start"
SKIP_CLOSE = "doc:no-check-end"

# Where a relative citation may be resolved from (repo root first, then the usual dirs).
# `doc` is included so a citation can point at a file sitting next to the docs.
SEARCH_DIRS = ("", "nanochat", "scripts", "tasks", "runs", "tests", "doc")


def iter_doc_files(doc_dir):
    for name in sorted(os.listdir(doc_dir)):
        if name.endswith(".md"):
            yield os.path.join(doc_dir, name)


def resolve(path, repo_root):
    """Map a citation path to a real file, or None."""
    for prefix in SEARCH_DIRS:
        candidate = os.path.join(repo_root, prefix, path) if prefix else os.path.join(repo_root, path)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


# -----------------------------------------------------------------------------
# Anchor resolution

def _target_names(target):
    """Flatten an assignment target into the bare names it binds.

    Handles `x = ...`, `x: int = ...`, and tuple unpacking such as
    `COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()`.
    """
    names = []
    if isinstance(target, ast.Name):
        names.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.extend(_target_names(elt))
    return names


def _python_anchors(lines):
    """Top-level and method defs/classes, module-level assigns, dataclass fields."""
    anchors = {}
    try:
        tree = ast.parse("\n".join(lines) + "\n")
    except SyntaxError:
        return anchors
    # module-level and class-level names
    def walk(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                anchors.setdefault(node.name, node.lineno)
                walk(node.body)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in _target_names(target):
                        anchors.setdefault(name, node.lineno)
            elif isinstance(node, ast.AnnAssign):
                for name in _target_names(node.target):
                    anchors.setdefault(name, node.lineno)
    walk(tree.body)
    # dataclass-style fields: `name: int = 2048` appear as AnnAssign at class level,
    # already covered above; also catch plain `name = value` inside classes via walk.
    return anchors


# A function/class does not "end" at a discoverable line without more AST work, so for
# `@Ln` assertions we resolve the *next* anchor and use it as the exclusive upper bound.
def _anchor_spans(lines, anchors):
    ordered = sorted(anchors.values())
    spans = {}
    for name, start in anchors.items():
        nxt = None
        for line_no in ordered:
            if line_no > start:
                nxt = line_no
                break
        spans[name] = (start, (nxt - 1) if nxt else len(lines))
    return spans


def _text_anchors(lines, path):
    """Anchors for non-python files: shell section comments, md headings."""
    anchors = {}
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if path.endswith(".sh"):
            m = re.match(r"^#\s*-{2,}\s*$", stripped)
            if m:
                continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if text:
                anchors.setdefault(text, i)
        if path.endswith(".md"):
            m = re.match(r"^#{1,6}\s+(.*?)\s*$", stripped)
            if m:
                anchors.setdefault(m.group(1), i)
    return anchors


def build_anchor_index(path, lines):
    if path.endswith(".py"):
        anchors = _python_anchors(lines)
        return anchors, _anchor_spans(lines, anchors)
    anchors = _text_anchors(lines, path)
    return anchors, {name: (start, len(lines)) for name, start in anchors.items()}


def find_anchor(anchors, name):
    """Exact match first, then the tail of a dotted name (`KVCache.prefill` -> `prefill`)."""
    if name in anchors:
        return name, anchors[name]
    tail = name.split(".")[-1].split()[-1]
    if tail in anchors:
        return tail, anchors[tail]
    return None, None


# -----------------------------------------------------------------------------
# Checking

class Result:
    def __init__(self):
        self.resolved = []
        self.errors = []
        self.warnings = []


def check_file(doc_path, repo_root, result):
    doc_name = os.path.basename(doc_path)
    skipping = False
    for line_no, line in enumerate(load_lines(doc_path), start=1):
        # Block form: `<!-- doc:no-check-start -->` ... `<!-- doc:no-check-end -->` lets a
        # section deliberately quote removed/invalid paths without failing the check.
        if SKIP_CLOSE in line:
            skipping = False
            continue
        if skipping:
            continue
        if SKIP_OPEN in line:
            skipping = True
            continue
        if SKIP_MARKER in line:
            continue
        matches = {}
        for pattern in (CITATION_RE, CITATION_RE_MD):
            for m in pattern.finditer(line):
                matches.setdefault(m.start(), m)
        for m in (matches[k] for k in sorted(matches)):
            path = m.group("path")
            anchor = m.group("anchor")
            anchor_line = m.group("anchor_line")
            start_line = m.group("line")
            end_line = m.group("line_end")
            quote = m.group("quote1") or m.group("quote2")
            where = f"{doc_name}:{line_no}"
            cite = m.group(0)

            real = resolve(path, repo_root)
            if real is None:
                result.errors.append(f"{where}: file not found: {path!r} (citation {cite!r})")
                continue

            src = load_lines(real)
            n = len(src)

            if anchor is not None:
                anchors, spans = build_anchor_index(real, src)
                found_name, found_line = find_anchor(anchors, anchor.strip())
                if found_line is None:
                    result.errors.append(
                        f"{where}: anchor not found: {path}#{anchor} (citation {cite!r})"
                    )
                    continue
                if anchor_line is not None:
                    want = int(anchor_line)
                    lo, hi = spans[found_name]
                    if not (lo <= want <= hi):
                        result.errors.append(
                            f"{where}: {path}#{found_name}@L{want} outside symbol span "
                            f"L{lo}-L{hi} (citation {cite!r})"
                        )
                        continue
                result.resolved.append(f"{where}: {path}#{found_name} -> L{found_line}")
                continue

            if start_line is not None:
                want = int(start_line)
                if want < 1 or want > n:
                    result.errors.append(
                        f"{where}: line {want} out of range for {path} ({n} lines) (citation {cite!r})"
                    )
                    continue
                if end_line is not None:
                    end = int(end_line)
                    if end < want or end > n:
                        result.errors.append(
                            f"{where}: line range {want}-{end} out of range for {path} ({n} lines)"
                        )
                        continue
                if quote:
                    text = src[want - 1]
                    if quote.strip() not in text:
                        result.errors.append(
                            f"{where}: {path}:{want} does not contain {quote.strip()!r} "
                            f"(line reads: {text.strip()[:70]!r})"
                        )
                        continue
                result.resolved.append(f"{where}: {path}:{want}")
                continue

            # bare file reference is always fine
            result.resolved.append(f"{where}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="list every resolved citation")
    parser.add_argument("--doc-dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = parser.parse_args()

    result = Result()
    for doc_path in iter_doc_files(args.doc_dir):
        check_file(doc_path, args.repo_root, result)

    if args.verbose:
        print("Resolved citations:")
        for entry in result.resolved:
            print(f"  OK   {entry}")
        print()

    print(f"Citations resolved: {len(result.resolved)}")
    if result.warnings:
        print(f"\nWARNINGS ({len(result.warnings)}):")
        for w in result.warnings:
            print(f"  WARN {w}")
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for e in result.errors:
            print(f"  FAIL {e}")
        print(
            "\nFix the citation (prefer `path#symbol` over bare line numbers), "
            "or mark the line with `doc:no-check` if it is a deliberate counter-example."
        )
        return 1

    print("All citations resolve. [OK]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
