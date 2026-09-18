"""
Keep the Chinese technical docs in doc/ honest about the source tree.

Every `file#symbol` / `file:line` citation in doc/*.md is resolved against the real
code by doc/check_citations.py. This module only wraps it so `pytest tests/` catches
stale references (renamed symbols, deleted modules, line numbers that drifted) instead
of letting them rot silently.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc"))

import check_citations  # noqa: E402  (path set up above on purpose)


def test_doc_citations_resolve():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_dir = os.path.join(repo_root, "doc")

    result = check_citations.Result()
    for doc_path in check_citations.iter_doc_files(doc_dir):
        check_citations.check_file(doc_path, repo_root, result)

    assert result.resolved, "no citations were found at all - did the docs move?"
    assert not result.errors, "stale doc citations:\n  " + "\n  ".join(result.errors)
