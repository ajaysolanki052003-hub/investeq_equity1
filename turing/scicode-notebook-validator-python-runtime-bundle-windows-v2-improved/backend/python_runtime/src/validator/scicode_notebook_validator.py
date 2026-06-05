#!/usr/bin/env python3

import argparse
import ast
import contextlib
import csv
import datetime
import io
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import coverage as coverage_lib  # pip install coverage
    HAS_COVERAGE = True
except ImportError:
    coverage_lib = None  # type: ignore[assignment]
    HAS_COVERAGE = False

LOG_FOLDER_DEFAULT = os.path.join("output", "validation_logs")
LOG_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_DRIVE_CREDENTIALS = "service_account.json"
SUMMARY_FOLDER_DEFAULT = os.path.join("output", "summary")
DRIVE_API_MAX_RETRIES = 5
DRIVE_API_RETRY_BASE_DELAY = 2  # seconds
COVERAGE_MIN_THRESHOLD = 50  # Minimum recommended test coverage percentage

# Function names that are known test-runner / test-executor patterns.
# Calls to these should never appear in the solution or testing template;
# the validator discovers and runs test_* functions automatically.
TEST_RUNNER_NAMES: frozenset = frozenset({
    "run", "run_all", "run_test", "run_tests",
    "run_all_tests", "run_all_test",
})

SECTION_REQUIRED_KEYS = [
    "prompt",
    "background",
    "testing template",
    "solution",
]

NOTEBOOK_SECTION_KEYS = [
    "prompt",
    "background",
    "testing template",
    "solution",
]

JSON_SECTION_REQUIRED_KEYS = [
    "testing template",
    "solution",
]


class ProgressTracker:
    """Simple progress tracker for batch processing."""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.start_time = datetime.datetime.now()
    
    def update(self, increment: int = 1) -> None:
        self.current += increment
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        progress_pct = (self.current / self.total * 100) if self.total > 0 else 0
        sys.stderr.write(
            f"\rProgress: {self.current}/{self.total} ({progress_pct:.1f}%) "
            f"[{remaining:.0f}s remaining]       "
        )
        sys.stderr.flush()
    
    def finish(self) -> None:
        elapsed = (datetime.datetime.now() - self.start_time).total_seconds()
        sys.stderr.write(f"\nCompleted in {elapsed:.1f}s\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Resume / checkpoint support
# ---------------------------------------------------------------------------

def _notebook_key(notebook: Path, root: Optional[Path]) -> str:
    if root is None:
        return notebook.name
    try:
        return notebook.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return notebook.name


def _is_worker_error(issues: List[str]) -> bool:
    joined = " ".join(issues).lower()
    return "worker timeout/error" in joined or "worker error" in joined


def save_checkpoint(
    per_notebook: Dict[Path, Tuple[List[str], List[str]]],
    checkpoint_path: Path,
    root: Optional[Path],
) -> None:
    data = {
        "root": str(root) if root is not None else None,
        "per_notebook": {
            _notebook_key(nb, root): {"issues": issues, "results": results}
            for nb, (issues, results) in per_notebook.items()
        },
    }
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    tmp_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
) -> Tuple[Dict[str, Tuple[List[str], List[str]]], Set[str]]:
    with checkpoint_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    per_notebook_raw = data.get("per_notebook", {})
    per_notebook: Dict[str, Tuple[List[str], List[str]]] = {}
    worker_error_keys: Set[str] = set()
    for key, val in per_notebook_raw.items():
        issues = list(val.get("issues", []))
        results = list(val.get("results", []))
        per_notebook[key] = (issues, results)
        if _is_worker_error(issues):
            worker_error_keys.add(key)
    return per_notebook, worker_error_keys


def is_test_failure_line(text: str) -> bool:
    return bool(re.match(r"^\s*FAIL\b", text))


def determine_status(issues: List[str], results: List[str]) -> str:
    """Determine notebook status.

    When tests were actually executed (results is non-empty), status is based
    solely on execution failures (lines starting with ``FAIL``).  Structural
    warnings (missing subsections, etc.) do NOT cause a FAIL when
    all tests pass.

    When no tests were executed, structural issues still cause FAIL.
    """
    failed_tests = collect_test_failures(issues, results)
    if results:  # tests were executed
        return "FAILED" if failed_tests else "SUCCESS"
    # structure-only mode: any issue => FAIL
    return "FAILED" if issues else "SUCCESS"


def collect_test_failures(issues: List[str], results: List[str]) -> List[str]:
    failures: List[str] = []
    seen: Set[str] = set()
    for entry in results + issues:
        if is_test_failure_line(entry) and entry not in seen:
            failures.append(entry)
            seen.add(entry)
    return failures


# ---------------------------------------------------------------------------
# Issue type registry
# ---------------------------------------------------------------------------
# Each key is an internal issue-type ID.
# "label"       – Short human-readable name shown in reports.
# "description" – One-line explanation of what this issue type means.
# "action"      – What the notebook author should do to fix it.
# ---------------------------------------------------------------------------
ISSUE_TYPES: Dict[str, Dict[str, str]] = {
    "STRUCTURE_CODE_PLACEMENT": {
        "label": "Structure: Code Cell Misplacement",
        "description": (
            "A code cell appears outside the allowed subsections "
            "(Testing Template or Solution)."
        ),
        "action": (
            "Move the code cell so it appears directly under one of the allowed "
            "subsection headings: '## Testing Template' or '## Solution' inside "
            "a '# Main Problem' or '# Subproblem N' section."
        ),
    },
    "STRUCTURE_INLINE_INSTALL": {
        "label": "Structure: Inline Package Install",
        "description": (
            "A code cell contains a pip/apt-get install command."
        ),
        "action": (
            "Remove the install command from the code cell. "
            "Dependencies should be declared in the Metadata section or a "
            "requirements file, not installed inline."
        ),
    },
    "STRUCTURE_MISSING_METADATA": {
        "label": "Structure: Missing Metadata Section",
        "description": (
            "The notebook is missing the top-level '# Metadata' section."
        ),
        "action": (
            "Add a '# Metadata' section containing problem metadata "
            "(field, subfield, difficulty level, etc.)."
        ),
    },
    "STRUCTURE_MISSING_TITLE": {
        "label": "Structure: Missing Title Section",
        "description": (
            "The notebook is missing the top-level '# Title' section."
        ),
        "action": (
            "Add a '# Title' section with a clear, descriptive title "
            "for the scientific problem."
        ),
    },
    "STRUCTURE_MISSING_SUBSECTION": {
        "label": "Structure: Missing Required Subsection",
        "description": (
            "A problem section is missing one of the required subsections "
            "(Prompt, Background, Testing Template, or Solution)."
        ),
        "action": (
            "Add the missing '## <Subsection>' heading and its content to "
            "the indicated section."
        ),
    },
    "STRUCTURE_NO_SECTIONS": {
        "label": "Structure: No Valid Sections Found",
        "description": (
            "The notebook does not contain the required problem sections "
            "(both '# Main Problem' and at least 2 '# Subproblem N' sections)."
        ),
        "action": (
            "Ensure the notebook has a '# Main Problem' heading AND at least 2 "
            "'# Subproblem N' headings (e.g. '# Subproblem 1', '# Subproblem 2') "
            "with the required subsections underneath."
        ),
    },
    "STRUCTURE_MISSING_MAIN_PROBLEM": {
        "label": "Structure: Missing Main Problem Section",
        "description": (
            "The notebook is missing the '# Main Problem' section."
        ),
        "action": (
            "Add a '# Main Problem' section with the required subsections "
            "(Prompt, Background, Testing Template, Solution)."
        ),
    },
    "STRUCTURE_INSUFFICIENT_SUBPROBLEMS": {
        "label": "Structure: Insufficient Subproblem Sections",
        "description": (
            "The notebook does not contain at least 2 '# Subproblem N' sections."
        ),
        "action": (
            "Ensure the notebook has at least 2 '# Subproblem N' headings "
            "(e.g. '# Subproblem 1', '# Subproblem 2') each with the required "
            "subsections (Prompt, Background, Testing Template, Solution)."
        ),
    },
    "TEST_NO_FUNCTIONS": {
        "label": "Test: No Test Functions Found",
        "description": (
            "The Testing Template does not contain any functions whose name "
            "starts with 'test_'."
        ),
        "action": (
            "Add at least one test function (e.g. def test_example():) in "
            "'## Testing Template' that validates the solution."
        ),
    },
    "TEST_MISSING_ASSERTIONS": {
        "label": "Test: Missing Assert Statements",
        "description": (
            "The test functions do not contain any 'assert' statements."
        ),
        "action": (
            "Add assert statements to the test functions to verify expected "
            "outputs (e.g. assert result == expected_value)."
        ),
    },
    "TEST_NOT_REFERENCING_FUNCTION": {
        "label": "Test: Tests Do Not Reference Declared Function",
        "description": (
            "The tests in Testing Template do not call/reference the function "
            "that the prompt asks the model to implement."
        ),
        "action": (
            "Update the test functions so they actually invoke and validate "
            "the target function declared in the prompt."
        ),
    },
    "TEST_FAILURE": {
        "label": "Test: Test Failure",
        "description": (
            "A test function raised an exception or assertion error."
        ),
        "action": (
            "Review the test logic and the solution implementation to fix "
            "the failing assertion or error."
        ),
    },
    "TEST_LOW_COVERAGE": {
        "label": "Test: Insufficient Test Coverage",
        "description": (
            "Static test-quality analysis found weak coverage signals in the "
            "testing template — e.g. target function is not exercised enough, "
            "assertion depth/variety is weak, edge-case diversity is low, or "
            "prompt constraints are not validated."
        ),
        "action": (
            "Strengthen test quality by adding diverse scenarios, richer "
            "assertions, edge-case tests, and explicit checks for prompt "
            "constraints. Prioritize behavior coverage over line-percentage."
        ),
    },
    "TEST_EXPLICIT_CALL": {
        "label": "Test: Explicit Test/Runner Function Call",
        "description": (
            "A test function (test_*) or runner function (run(), run_all(), "
            "run_tests(), etc.) is explicitly called in the solution or testing "
            "template code. Test functions should only be defined — the "
            "validator discovers and runs them automatically."
        ),
        "action": (
            "Remove explicit calls to test functions (e.g. test_foo()) and "
            "runner functions (e.g. run(), run_all(), run_tests()) from the "
            "Solution and Testing Template code. Only define test functions "
            "with 'def test_*():' — the validator's test runner will discover "
            "and execute them automatically."
        ),
    },
    "COMPILATION_SETUP_ERROR": {
        "label": "Compilation: Setup / Execution Error",
        "description": (
            "The solution or test code raised an error during execution "
            "(syntax error, missing import, undefined variable, etc.)."
        ),
        "action": (
            "Fix the error in the Solution and/or Testing Template code cells. "
            "Check for syntax errors, missing imports, or undefined variables."
        ),
    },
    "IMPL_MISSING_FUNCTION": {
        "label": "Implementation: Missing Function in Solution",
        "description": (
            "A function expected by the prompt is not implemented "
            "in the Solution code cell."
        ),
        "action": (
            "Add the missing function implementation to the '## Solution' code cell."
        ),
    },
    "LATEX_FORMATTING": {
        "label": "LaTeX: Formatting Issue",
        "description": (
            "The Prompt or Background section contains malformed LaTeX — "
            "e.g. unmatched delimiters ($, $$, \\(, \\[), unbalanced braces, "
            "unmatched \\begin/\\end environments, or broken commands."
        ),
        "action": (
            "Fix the LaTeX formatting in the indicated section: ensure every "
            "opening delimiter has a matching close, every '{' has a '}', "
            "every \\begin{env} has a \\end{env}, and commands like \\frac, "
            "\\sqrt, \\mathbb are followed by '{…}'."
        ),
    },
    "OTHER": {
        "label": "Other",
        "description": "An issue that does not match any known category.",
        "action": "Review and fix this issue in the notebook.",
    },
    "SOLUTION_CONTAINS_TEST_ARTIFACTS": {
        "label": "Solution: Test Artifacts in Solution Code",
        "description": (
            "The Solution code cell includes test code or test-only constructs."
        ),
        "action": (
            "Move test cases, assertions, test_* functions/calls, or __name__ guards "
            "into the '## Testing Template' section. The Solution should only contain "
            "the implementation."
        ),
    },
    "TITLE_INVALID_CONTENT": {
        "label": "Structure: Invalid Title Content",
        "description": (
            "The cell immediately after '# Title' contains non-title content "
            "such as URLs/links, code, LaTeX blocks, or excessive text."
        ),
        "action": (
            "Replace the title cell content with a single concise line of "
            "plain text describing the scientific problem."
        ),
    },
    "PROMPT_MISSING_FUNCTION_SIGNATURE": {
        "label": "Structure: Prompt Missing Function Signature",
        "description": (
            "The Prompt section does not contain a fenced code block with a "
            "function signature (def function_name(...))."
        ),
        "action": (
            "Add a fenced code block (``` ```) to the Prompt containing the "
            "function signature stub (def, docstring, return) that the model "
            "should implement."
        ),
    },
    "PROMPT_MULTIPLE_FUNCTIONS": {
        "label": "Structure: Prompt Has Multiple Function Definitions",
        "description": (
            "The Prompt code fence contains multiple function definitions. "
            "Each prompt should declare exactly one function signature — "
            "helper functions belong in the Solution cell."
        ),
        "action": (
            "Keep only the main function signature in the Prompt code fence. "
            "Move any helper function definitions to the ## Solution cell."
        ),
    },
    "MISSING_IMPORTS": {
        "label": "Structure: Missing Import Statements",
        "description": (
            "The solution or test code uses modules/libraries (e.g. numpy, "
            "scipy, torch) but does not include the corresponding import "
            "statements."
        ),
        "action": (
            "Add the missing import statements to the top of the ## Solution "
            "cell. For example: 'import numpy as np', 'import scipy', etc."
        ),
    },
}


def classify_issue(issue: str) -> str:
    """Return the issue-type ID for a given issue message."""
    il = issue.lower()

    if "code cell" in il and "outside" in il:
        return "STRUCTURE_CODE_PLACEMENT"
    if "inline" in il and "install" in il:
        return "STRUCTURE_INLINE_INSTALL"
    if "missing" in il and "'# metadata'" in il:
        return "STRUCTURE_MISSING_METADATA"
    if "missing" in il and "'# title'" in il:
        return "STRUCTURE_MISSING_TITLE"
    if "missing required subsection" in il or ("missing" in il and "section" in il and "subsection" in il):
        return "STRUCTURE_MISSING_SUBSECTION"
    if "no valid problem sections" in il:
        return "STRUCTURE_NO_SECTIONS"
    if "missing '# main problem' section" in il:
        return "STRUCTURE_MISSING_MAIN_PROBLEM"
    if "subproblem section(s), but at least 2 are required" in il:
        return "STRUCTURE_INSUFFICIENT_SUBPROBLEMS"
    if "no test functions" in il or "no callable test functions" in il:
        return "TEST_NO_FUNCTIONS"
    if "assert" in il and ("missing" in il or "no " in il):
        return "TEST_MISSING_ASSERTIONS"
    if "does not reference" in il or "do not reference" in il:
        return "TEST_NOT_REFERENCING_FUNCTION"
    if il.startswith("fail") or "raised" in il and "test did not pass" in il:
        return "TEST_FAILURE"
    if "low test coverage" in il:
        return "TEST_LOW_COVERAGE"
    if "low test quality score" in il:
        return "TEST_LOW_COVERAGE"
    if "insufficient assertion depth" in il:
        return "TEST_LOW_COVERAGE"
    if "edge-case coverage appears weak" in il:
        return "TEST_LOW_COVERAGE"
    if "tests never call the target function" in il:
        return "TEST_LOW_COVERAGE"
    if "called only" in il and "time(s) in tests" in il:
        return "TEST_LOW_COVERAGE"
    if "only one assertion pattern" in il:
        return "TEST_LOW_COVERAGE"
    if "prompt mentions constraints that tests may not cover" in il:
        return "TEST_LOW_COVERAGE"
    if "explicitly calls test function" in il:
        return "TEST_EXPLICIT_CALL"
    if "explicitly calls runner function" in il:
        return "TEST_EXPLICIT_CALL"
    if "setup failed" in il or "code setup failed" in il:
        return "COMPILATION_SETUP_ERROR"
    if "solution code contains test artifacts" in il:
        return "SOLUTION_CONTAINS_TEST_ARTIFACTS"
    if "title content issue" in il:
        return "TITLE_INVALID_CONTENT"
    if "prompt missing function signature" in il or "prompt has no fenced code block" in il:
        return "PROMPT_MISSING_FUNCTION_SIGNATURE"
    if "prompt code fence contains multiple function definitions" in il:
        return "PROMPT_MULTIPLE_FUNCTIONS"
    if "no corresponding import" in il:
        return "MISSING_IMPORTS"
    if "not implemented" in il or ("declared" in il and "not implemented" in il):
        return "IMPL_MISSING_FUNCTION"
    if "is declared" in il and "not implemented" in il:
        return "IMPL_MISSING_FUNCTION"
    # Fallback patterns for function-missing-in-solution messages
    if "function" in il and "not implemented" in il:
        return "IMPL_MISSING_FUNCTION"
    # LaTeX formatting issues
    if "latex" in il and ("unmatched" in il or "not followed by" in il or "formatting" in il):
        return "LATEX_FORMATTING"
    if "unmatched" in il and any(
        kw in il for kw in ("delimiter", "\\begin", "\\end", "brace", "math")
    ):
        return "LATEX_FORMATTING"

    return "OTHER"


def get_action_for_issue(issue: str) -> str:
    """Return a human-readable action item for a given issue message."""
    issue_type = classify_issue(issue)
    return ISSUE_TYPES[issue_type]["action"]


def get_issue_type_label(issue: str) -> str:
    """Return the human-readable issue-type label for a given issue message."""
    issue_type = classify_issue(issue)
    return ISSUE_TYPES[issue_type]["label"]


# ---------------------------------------------------------------------------
# Unified logging: one .log and one .csv per run
# ---------------------------------------------------------------------------

def format_notebook_block(
    notebook: Path,
    issues: List[str],
    results: List[str],
) -> List[str]:
    """Format the log block for a single notebook (for merging into existing reports)."""
    status = determine_status(issues, results)
    fail_count = len(collect_test_failures(issues, results))
    lines = [
        f"{'='*70}",
        f"  [{status}]  {notebook.name}",
        f"  Issues: {len(issues)}  |  Test Failures: {fail_count}",
        f"{'='*70}",
    ]
    section_data = _build_section_summary(issues, results)
    if section_data:
        for sec_name, sec_info in section_data.items():
            if sec_name == "main problem":
                label = "Main Function"
            elif sec_name == "General":
                label = "General / Notebook-Level"
            else:
                label = f"Sub-Function ({sec_name})"
            sec_status = (
                "PASS" if sec_info["fail"] == 0 and sec_info["pass"] > 0
                else "FAIL" if sec_info["fail"] > 0
                else "WARN" if sec_info["warnings"]
                else "----"
            )
            lines.append(
                f"  [{sec_status}] {label}  "
                f"(pass={sec_info['pass']}, fail={sec_info['fail']}, "
                f"warnings={len(sec_info['warnings'])})"
            )
            for entry in sec_info["pass_details"]:
                lines.append(f"         PASS  {entry}")
            for entry in sec_info["fail_details"]:
                lines.append(f"         FAIL  {entry}")
            for entry in sec_info["warnings"]:
                lines.append(f"         WARN  {entry}")
    elif not issues:
        lines.append("  No issues found.")
    lines.append("")
    return lines


def write_run_log(
    per_notebook: Dict[Path, Tuple[List[str], List[str]]],
    log_folder: Path,
    source_label: str,
    log_path: Optional[Path] = None,
) -> Path:
    """Write a single consolidated .log file for the entire run.

    Combines what was previously spread across per-notebook logs, batch
    issues log, failures log, stats file, and comprehensive report into
    one file.
    """
    timestamp = datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
    log_folder.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = log_folder / f"{timestamp}_summary.log"

    total = len(per_notebook)
    passed = sum(
        1 for issues, results in per_notebook.values()
        if determine_status(issues, results) == "SUCCESS"
    )
    failed = total - passed
    total_issues = sum(len(issues) for issues, _ in per_notebook.values())
    total_failures = sum(
        len(collect_test_failures(issues, results))
        for issues, results in per_notebook.values()
    )

    lines = [
        "=" * 80,
        "  VALIDATION REPORT",
        "=" * 80,
        f"  Generated : {timestamp}",
        f"  Source    : {source_label}",
        f"  Notebooks : {total}",
        f"  Passed    : {passed} ({passed/total*100:.1f}%)" if total else "  Passed    : 0",
        f"  Failed    : {failed} ({failed/total*100:.1f}%)" if total else "  Failed    : 0",
        f"  Issues    : {total_issues}",
        f"  Test fails: {total_failures}",
        "=" * 80,
        "",
    ]

    # Per-notebook detail
    for notebook in sorted(per_notebook.keys(), key=lambda p: p.name):
        issues, results = per_notebook[notebook]
        lines.extend(format_notebook_block(notebook, issues, results))

    lines.extend(["=" * 80, "  END OF REPORT", "=" * 80])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def notebook_csv_row(
    notebook: Path,
    issues: List[str],
    results: List[str],
) -> Dict[str, any]:
    """Build the CSV row for a single notebook (for merging into existing reports)."""
    status = determine_status(issues, results)
    fail_count = len(collect_test_failures(issues, results))
    sec_summary = _build_section_summary(issues, results)
    solution_pass = sum(
        1 for s in sec_summary.values()
        if "solution_execution" in " ".join(s["pass_details"])
    )
    solution_fail = sum(
        1 for s in sec_summary.values()
        if "solution_execution" in " ".join(s["fail_details"])
    )
    test_pass = sum(s["pass"] for s in sec_summary.values()) - solution_pass
    test_fail = sum(s["fail"] for s in sec_summary.values()) - solution_fail
    section_labels = "; ".join(
        f"{k}:{'PASS' if v['fail']==0 and v['pass']>0 else 'FAIL' if v['fail']>0 else '----'}"
        for k, v in sec_summary.items()
    )
    return {
        "Notebook": notebook.stem,
        "Status": status,
        "SolutionPass": solution_pass,
        "SolutionFail": solution_fail,
        "TestPass": test_pass,
        "TestFail": test_fail,
        "IssueCount": len(issues),
        "FailCount": fail_count,
        "SectionResults": section_labels,
    }


def write_run_csv(
    per_notebook: Dict[Path, Tuple[List[str], List[str]]],
    log_folder: Path,
    csv_path: Optional[Path] = None,
) -> Path:
    """Write a single consolidated .csv summary for the entire run."""
    timestamp = datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
    log_folder.mkdir(parents=True, exist_ok=True)
    if csv_path is None:
        csv_path = log_folder / f"{timestamp}_summary.csv"

    rows = []
    for notebook in sorted(per_notebook.keys(), key=lambda p: p.name):
        issues, results = per_notebook[notebook]
        rows.append(notebook_csv_row(notebook, issues, results))

    fieldnames = [
        "Notebook", "Status",
        "SolutionPass", "SolutionFail",
        "TestPass", "TestFail",
        "IssueCount", "FailCount",
        "SectionResults",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _build_section_summary(
    issues: List[str], results: List[str]
) -> Dict[str, Dict[str, any]]:
    """Build a per-section (subproblem / main problem) summary from issues and results.

    All issues are attributed to a section where possible.  Issues that
    cannot be mapped to a specific section (e.g. missing Metadata /
    Title, notebook-level structural problems) are collected under the
    special ``"General"`` section so they still appear in the report.
    """
    sections: Dict[str, Dict[str, any]] = {}

    def _ensure_section(name: str) -> None:
        if name not in sections:
            sections[name] = {
                "pass": 0, "fail": 0,
                "pass_details": [], "fail_details": [], "warnings": [],
            }

    for entry in results:
        # e.g. "PASS main problem: solution_execution"
        m = re.match(r"^PASS\s+(.+?):\s+(.+)$", entry)
        if m:
            sec = m.group(1).strip()
            detail = m.group(2).strip()
            _ensure_section(sec)
            sections[sec]["pass"] += 1
            sections[sec]["pass_details"].append(detail)

    for entry in issues:
        m_fail = re.match(r"^FAIL\s+(.+?):\s+(.+)$", entry)
        if m_fail:
            sec = m_fail.group(1).strip()
            detail = m_fail.group(2).strip()
            _ensure_section(sec)
            sections[sec]["fail"] += 1
            sections[sec]["fail_details"].append(detail)
        else:
            # Warnings — try to identify the section from the text
            m_sec = re.search(r"section '([^']+)'", entry, re.IGNORECASE)
            if m_sec:
                sec = m_sec.group(1).strip()
                _ensure_section(sec)
                sections[sec]["warnings"].append(entry)
            else:
                # General / notebook-level issue (metadata, title, etc.)
                _ensure_section("General")
                sections["General"]["warnings"].append(entry)

    # Sort so subproblems come first (numerically), then main problem, then General last
    def _sort_key(item):
        name = item[0]
        if name.startswith("subproblem"):
            m = re.search(r"(\d+)", name)
            return (0, int(m.group(1)) if m else 9999, name)
        if name == "main problem":
            return (1, 0, name)
        if name == "General":
            return (3, 0, name)
        return (2, 0, name)

    return dict(sorted(sections.items(), key=_sort_key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate notebook tests against declared functions."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to a single .ipynb file.",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        help="Folder containing .ipynb files.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Path to a SciCode JSON file.",
    )
    parser.add_argument(
        "--input-json-folder",
        type=Path,
        help="Folder containing SciCode JSON files.",
    )
    parser.add_argument(
        "--drive-folder-url",
        type=str,
        help="Google Drive folder URL containing .ipynb files.",
    )
    parser.add_argument(
        "--drive-credentials",
        type=Path,
        default=Path(DEFAULT_DRIVE_CREDENTIALS),
        help="Path to the Google service account JSON credentials.",
    )
    parser.add_argument(
        "--drive-batch-size",
        type=int,
        help="Max number of Drive notebooks to validate per run.",
    )
    parser.add_argument(
        "--drive-batch-start",
        type=int,
        default=0,
        help="0-based offset for Drive notebook batches.",
    )
    parser.add_argument(
        "--subfolder-access",
        action="store_true",
        help="(Deprecated: subfolders are now included by default) Include notebooks inside Drive subfolders.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run successive Drive batches until completion.",
    )
    parser.add_argument(
        "--log-folder",
        type=Path,
        default=Path(LOG_FOLDER_DEFAULT),
        help="Folder to store log reports.",
    )
    parser.add_argument(
        "--no-run-test",
        action="store_true",
        help="Skip executing test functions in the notebook.",
    )
    parser.add_argument(
        "--assert-checks",
        action="store_true",
        help="Require assert statements in test templates.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output from notebook test runs.",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-notebook timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint JSON created by a prior run.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Explicit checkpoint JSON path (default: <log_folder>/<timestamp>_checkpoint.json).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint every N notebooks (default: 1).",
    )
    return parser.parse_args()


def normalize_heading(text: str) -> str:
    return text.strip().lower()


def join_markdown(source: List[str]) -> str:
    return "".join(source).strip()


def join_code(source: List[str]) -> str:
    return "".join(source).rstrip()


def collect_notebooks(folder: Path) -> List[Path]:
    return sorted(path for path in folder.glob("*.ipynb") if path.is_file())


def extract_drive_folder_id(folder_url: str) -> str:
    parsed = urlparse(folder_url)
    if parsed.path.startswith("/drive/folders/"):
        return parsed.path.split("/drive/folders/")[-1].split("/")[0]
    query = parse_qs(parsed.query)
    folder_ids = query.get("id")
    if folder_ids:
        return folder_ids[0]
    raise ValueError("Unable to extract Google Drive folder ID from URL.")


def ensure_drive_deps() -> None:
    try:
        import google.oauth2.service_account  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
        import google_auth_httplib2 # noqa: F401
        import httplib2 # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Google Drive support requires google-api-python-client. "
            "Install with: pip install google-api-python-client google-auth-httplib2 httplib2"
        ) from exc


def build_drive_service(credentials_path: Path):
    ensure_drive_deps()
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    import google_auth_httplib2
    import httplib2

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Google Drive credentials not found: {credentials_path}"
        )

    credentials = Credentials.from_service_account_file(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

    # Create an HTTP transport with a generous timeout (10 minutes),
    # then wrap it with the service account credentials.
    # Pass ONLY http= to build() (credentials= and http= are mutually exclusive).
    http = httplib2.Http(timeout=600)
    authorized_http = google_auth_httplib2.AuthorizedHttp(credentials, http=http)

    return build("drive", "v3", http=authorized_http)


def _drive_api_call_with_retry(callable_fn, description: str = "API call"):
    """Execute a Google Drive API call with exponential backoff retry.

    Retries on transient errors: TimeoutError, ConnectionError, OSError,
    and HTTP 429/5xx responses.
    """
    for attempt in range(1, DRIVE_API_MAX_RETRIES + 1):
        try:
            return callable_fn()
        except (TimeoutError, ConnectionError, OSError) as exc:
            if attempt == DRIVE_API_MAX_RETRIES:
                raise
            delay = DRIVE_API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(
                f"  Retry {attempt}/{DRIVE_API_MAX_RETRIES} for {description} "
                f"({type(exc).__name__}), waiting {delay}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
        except Exception as exc:
            # Retry on HTTP 429 (rate limit) or 5xx (server error)
            exc_str = str(exc)
            if any(code in exc_str for code in ("429", "500", "502", "503", "504")):
                if attempt == DRIVE_API_MAX_RETRIES:
                    raise
                delay = DRIVE_API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(
                    f"  Retry {attempt}/{DRIVE_API_MAX_RETRIES} for {description} "
                    f"(HTTP error), waiting {delay}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                raise


def scan_drive_folder(
    folder_url: str,
    credentials_path: Path,
    include_subfolders: bool = True,
) -> Dict[str, any]:
    """
    Pre-flight scan to count notebooks and folder structure.
    Returns statistics about the Drive folder contents.
    """
    folder_id = extract_drive_folder_id(folder_url)
    service = build_drive_service(credentials_path)

    queue = [folder_id]
    total_items = 0
    total_notebooks = 0
    total_folders = 0

    while queue:
        current_folder = queue.pop(0)
        page_token: Optional[str] = None

        while True:
            query = f"'{current_folder}' in parents and trashed = false"
            response = _drive_api_call_with_retry(
                lambda: service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, shortcutDetails)",
                    pageToken=page_token,
                    orderBy="name",
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute(),
                description=f"scan folder {current_folder}",
            )

            for entry in response.get("files", []):
                file_id = entry.get("id")
                file_name = entry.get("name") or "notebook.ipynb"
                mime_type = entry.get("mimeType")
                if not file_id:
                    continue

                total_items += 1

                if mime_type == "application/vnd.google-apps.shortcut":
                    shortcut = entry.get("shortcutDetails") or {}
                    file_id = shortcut.get("targetId")
                    mime_type = shortcut.get("targetMimeType")
                    if not file_id:
                        continue

                if include_subfolders and mime_type == "application/vnd.google-apps.folder":
                    total_folders += 1
                    queue.append(file_id)
                    continue

                is_notebook = mime_type == "application/vnd.google-apps.notebook" or file_name.endswith(
                    ".ipynb"
                )
                if is_notebook:
                    total_notebooks += 1

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return {
        "total_items": total_items,
        "total_notebooks": total_notebooks,
        "total_folders": total_folders,
    }


def download_drive_folder(
    folder_url: str,
    credentials_path: Path,
    batch_start: int = 0,
    batch_size: Optional[int] = None,
    include_subfolders: bool = True,
) -> Tuple[Path, int]:
    folder_id = extract_drive_folder_id(folder_url)
    service = build_drive_service(credentials_path)
    output_dir = Path(tempfile.mkdtemp(prefix="scicode_drive_"))

    queue = [folder_id]
    files: List[Path] = []
    fetched = 0
    kept = 0
    batch_limit_reached = False
    all_items_count = 0  # Track total items found
    notebook_count = 0   # Track notebooks found
    folder_count = 0     # Track subfolders found

    while queue and not batch_limit_reached:
        current_folder = queue.pop(0)
        page_token: Optional[str] = None

        while True:
            query = f"'{current_folder}' in parents and trashed = false"
            response = _drive_api_call_with_retry(
                lambda: service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, shortcutDetails)",
                    pageToken=page_token,
                    orderBy="name",
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute(),
                description=f"list folder {current_folder}",
            )

            for entry in response.get("files", []):
                file_id = entry.get("id")
                file_name = entry.get("name") or "notebook.ipynb"
                mime_type = entry.get("mimeType")
                if not file_id:
                    continue

                all_items_count += 1

                if mime_type == "application/vnd.google-apps.shortcut":
                    shortcut = entry.get("shortcutDetails") or {}
                    file_id = shortcut.get("targetId")
                    mime_type = shortcut.get("targetMimeType")
                    if not file_id:
                        continue

                if include_subfolders and mime_type == "application/vnd.google-apps.folder":
                    folder_count += 1
                    queue.append(file_id)
                    continue

                is_notebook = mime_type == "application/vnd.google-apps.notebook" or file_name.endswith(
                    ".ipynb"
                )
                if not is_notebook:
                    continue

                notebook_count += 1

                if fetched < batch_start:
                    fetched += 1
                    continue

                if batch_size is not None and kept >= batch_size:
                    batch_limit_reached = True
                    break

                fetched += 1
                if mime_type == "application/vnd.google-apps.notebook":
                    if not file_name.endswith(".ipynb"):
                        file_name = f"{file_name}.ipynb"
                    dl_request = service.files().export_media(
                        fileId=file_id,
                        mimeType="application/x-ipynb+json",
                    )
                else:
                    dl_request = service.files().get_media(
                        fileId=file_id,
                        supportsAllDrives=True,
                    )

                target_path = output_dir / file_name
                # Avoid overwriting files with same name from different folders
                if target_path.exists():
                    target_path = output_dir / f"{file_id}_{file_name}"

                content = _drive_api_call_with_retry(
                    dl_request.execute,
                    description=f"download {file_name}",
                )
                with target_path.open("wb") as output_file:
                    output_file.write(content)
                files.append(target_path)
                kept += 1

                # Progress output every 10 files
                if kept % 10 == 0:
                    print(f"  Downloaded {kept} notebooks so far...", file=sys.stderr)

            if batch_limit_reached:
                break
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    if not files:
        debug_info = f"(Found {all_items_count} total items, {notebook_count} notebooks, {folder_count} subfolders)"
        error_msg = (
            f"No .ipynb notebooks found in Drive folder. {debug_info}\n"
            "Possible causes:\n"
            "  1. The folder is empty\n"
            "  2. The folder contains no .ipynb files\n"
            "  3. Your service account doesn't have read access to the folder\n"
            "  4. The folder ID is incorrect\n\n"
            "Check your credentials in service_account.json and verify the folder URL is correct."
        )
        raise FileNotFoundError(error_msg)

    return output_dir, kept


def extract_functions(code: str, top_level_only: bool = True) -> List[str]:
    """Extract function names from code.
    
    If top_level_only is True (default), only returns functions defined at
    indentation level 0 — i.e. true top-level defs. Nested/local helper
    functions inside other functions are excluded to avoid false positives
    when checking test references.
    """
    names = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and "(" in stripped:
            if top_level_only:
                # Only count defs at column 0 (no leading whitespace)
                if line.startswith("def "):
                    name = stripped[4:].split("(")[0].strip()
                    if name:
                        names.append(name)
            else:
                name = stripped[4:].split("(")[0].strip()
                if name:
                    names.append(name)
    return names


def extract_test_functions(code: str) -> List[str]:
    return [name for name in extract_functions(code, top_level_only=True) if name.startswith("test_")]


def extract_test_function_bodies(code: str) -> List[Tuple[str, str]]:
    """Extract individual test function bodies from a testing template.

    Returns a list of (function_name, function_body) tuples where
    function_body includes the full ``def test_...`` block.
    """
    lines = code.splitlines()
    test_starts: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        if line.startswith("def test") and "(" in line:
            name = line.strip()[4:].split("(")[0].strip()
            test_starts.append((i, name))
    if not test_starts:
        return []

    results: List[Tuple[str, str]] = []
    for idx, (start, name) in enumerate(test_starts):
        # End is next test function start, or next top-level def, or end of file
        end = len(lines)
        for j in range(start + 1, len(lines)):
            stripped = lines[j].lstrip()
            # Another top-level def (no indent) means end of this function
            if stripped.startswith("def ") and len(lines[j]) - len(stripped) == 0:
                end = j
                break
        body = "\n".join(lines[start:end])
        results.append((name, body))
    return results


def has_procedural_tests(code: str) -> bool:
    """Detect procedural/inline test style — code that runs assertions or
    prints PASS/FAIL at module level without wrapping in test_ functions.
    """
    has_assertions = bool(re.findall(r"^\s*assert\b", code, re.MULTILINE))
    has_pass_fail = bool(re.search(r"(?:PASS|FAIL|✔|✘)", code))
    return has_assertions or has_pass_fail


def detect_solution_test_duplication(solution: str, test_template: str) -> bool:
    """Detect if the solution cell contains duplicated test code from the
    testing template. This happens when test code is copy-pasted into
    the solution cell.
    """
    if not solution or not test_template:
        return False
    # Check if test_ functions from the template are also in the solution
    test_funcs_in_template = extract_test_functions(test_template)
    test_funcs_in_solution = extract_test_functions(solution)
    if not test_funcs_in_template:
        return False
    overlap = set(test_funcs_in_template) & set(test_funcs_in_solution)
    # If significant overlap (>50% of template test functions are in solution)
    if len(overlap) >= max(1, len(test_funcs_in_template) // 2):
        return True
    # Also check if large blocks of test template text appear in solution
    template_lines = [l.strip() for l in test_template.splitlines() if l.strip()]
    solution_lines = [l.strip() for l in solution.splitlines() if l.strip()]
    if len(template_lines) < 5:
        return False
    matched = sum(1 for l in template_lines if l in solution_lines)
    return matched > len(template_lines) * 0.5


def count_assertions(code: str) -> int:
    return len(re.findall(r"\bassert\b", code))


def detect_solution_test_artifacts(code: str) -> List[str]:
    issues: List[str] = []
    if not code:
        return issues

    if re.search(r"\bassert\b", code):
        issues.append("contains 'assert' statements")
    if re.search(r"^\s*def\s+test_[A-Za-z0-9_]*\s*\(", code, flags=re.MULTILINE):
        issues.append("defines test_ functions")
    if re.search(r"\bpytest\b", code):
        issues.append("references pytest")
    if re.search(r"\bunittest\b", code):
        issues.append("references unittest")
    if re.search(r"\bTestCase\b", code):
        issues.append("uses unittest.TestCase")
    if re.search(r"\bassertRaises\b", code):
        issues.append("uses unittest assertions")
    if "__name__" in code:
        issues.append("uses __name__ guard")

    # AST-based: detect explicit calls to test_*() and runner functions
    test_func_names = [
        n for n in extract_functions(code, top_level_only=True)
        if n.startswith("test_")
    ]
    called_tests = detect_test_function_calls(code, test_func_names)
    if called_tests:
        issues.append(f"calls test function(s): {', '.join(called_tests)}")

    called_runners = detect_test_function_calls(code, list(TEST_RUNNER_NAMES))
    if called_runners:
        issues.append(
            f"calls runner function(s): {', '.join(n + '()' for n in called_runners)}"
        )

    return issues


def detect_test_function_calls(code: str, test_func_names: List[str]) -> List[str]:
    """Detect explicit calls to ``test_*()`` functions in *code* using AST.

    Only reports actual **call** expressions (``test_foo(...)``).  Function
    *definitions* (``def test_foo(...):``) are not flagged.  Returns a
    deduplicated list of test function names that are called.
    """
    if not code or not test_func_names:
        return []

    test_set = set(test_func_names)
    called: List[str] = []
    seen: Set[str] = set()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback: regex that skips lines starting with ``def ``
        for name in test_func_names:
            pattern = rf"(?<!def\s){re.escape(name)}\s*\("
            if re.search(pattern, code):
                called.append(name)
        return called

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_id: Optional[str] = None
            if isinstance(node.func, ast.Name):
                func_id = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_id = node.func.attr
            if func_id and func_id in test_set and func_id not in seen:
                called.append(func_id)
                seen.add(func_id)

    return called


def detect_test_template_artifacts(code: str) -> List[str]:
    """Detect test-runner boilerplate that should not appear in the testing template.

    The testing template should define individual ``test_*`` functions that
    the validator can discover and call independently.  Patterns like
    ``run_all_tests()`` with a ``__name__`` guard bypass the validator's
    test-discovery mechanism and must be flagged.
    """
    issues: List[str] = []
    if not code:
        return issues

    # Defining a runner function that aggregates tests
    if re.search(r"^\s*def\s+run_all_tests?\s*\(", code, flags=re.MULTILINE):
        issues.append("defines 'run_all_tests()' runner")
    if re.search(r"^\s*def\s+run_tests?\s*\(", code, flags=re.MULTILINE):
        issues.append("defines 'run_tests()' runner")
    if re.search(r"^\s*def\s+run_all\s*\(", code, flags=re.MULTILINE):
        issues.append("defines 'run_all()' runner")
    if re.search(r"^\s*def\s+run\s*\(", code, flags=re.MULTILINE):
        issues.append("defines 'run()' runner")

    # __name__ guard that invokes the runner
    if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', code):
        issues.append("uses '__name__' guard")

    # Manual iteration over test functions (e.g., for fn in [test1, test2]: fn())
    if re.search(
        r"for\s+\w+\s*,?\s*\w*\s+in\s+(?:enumerate\s*\()?\s*\["
        r"\s*test_",
        code,
    ):
        issues.append("iterates over test list manually")

    # AST-based: detect explicit calls to test_*() and runner functions
    test_func_names = [
        n for n in extract_functions(code, top_level_only=True)
        if n.startswith("test_")
    ]
    called_tests = detect_test_function_calls(code, test_func_names)
    if called_tests:
        issues.append(
            f"calls test function(s): {', '.join(n + '()' for n in called_tests)}"
        )

    called_runners = detect_test_function_calls(code, list(TEST_RUNNER_NAMES))
    if called_runners:
        issues.append(
            f"calls runner function(s): {', '.join(n + '()' for n in called_runners)}"
        )

    return issues


def validate_prompt_function_signature(prompt_text: str) -> List[str]:
    """Check that the Prompt section contains a fenced code block with a function signature.

    A well-formed prompt should include a ``` ``` block containing at least
    one ``def function_name(...)`` stub so the model knows what to implement.
    The stub should also include a docstring and a return statement.
    Returns a list of issues (empty if the prompt is valid).
    """
    issues: List[str] = []
    if not prompt_text or not prompt_text.strip():
        return issues  # missing-prompt is caught separately by validate_section_keys

    # Extract all fenced code blocks
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", prompt_text, re.DOTALL)

    if not code_blocks:
        issues.append("prompt has no fenced code block (expected ``` ``` with function signature)")
        return issues

    # Check if any code block contains a function definition
    blocks_with_func = [
        block for block in code_blocks
        if re.search(r"^\s*def\s+\w+\s*\(", block, re.MULTILINE)
    ]
    if not blocks_with_func:
        issues.append(
            "prompt missing function signature — code block(s) found but none "
            "contain a 'def function_name(...)' stub"
        )
        return issues

    # Check for multiple code fences containing function definitions
    if len(blocks_with_func) > 1:
        issues.append(
            "prompt has multiple code fences with function definitions — "
            "merge all function stubs into a single fenced code block "
            "so the JSON generator can extract them correctly"
        )

    # Combine all function-bearing blocks for further checks
    combined = "\n\n".join(blocks_with_func)

    # Check for multiple function definitions within the code fence(s).
    # Each prompt should define exactly ONE function signature; helper
    # functions belong in the Solution cell, not the Prompt.
    all_defs = re.findall(r"^\s*def\s+(\w+)\s*\(", combined, re.MULTILINE)
    if len(all_defs) > 1:
        defs_str = ", ".join(f"'{d}()'" for d in all_defs)
        issues.append(
            f"prompt code fence contains multiple function definitions: {defs_str} — "
            f"each prompt should declare exactly one function signature. "
            f"Move helper functions to the ## Solution cell."
        )

    # Check for docstring in the code fence
    has_docstring = bool(
        re.search(r'(\"\"\".*?\"\"\"|\'\'\'.*?\'\'\')', combined, re.DOTALL)
    )
    if not has_docstring:
        issues.append(
            "prompt code fence missing docstring — the function stub should include "
            "a triple-quoted docstring describing inputs, outputs, and return type"
        )

    # Check for return statement in the code fence
    has_return = any(
        line.strip().startswith("return ")
        or line.strip() == "return"
        for line in combined.splitlines()
    )
    # Also accept stubs that only have pass/... as body (no return needed for stubs)
    body_lines = [
        line.strip() for line in combined.splitlines()
        if line.strip()
        and not line.strip().startswith("def ")
        and not line.strip().startswith("#")
        and not line.strip().startswith('"""')
        and not line.strip().startswith("'''")
    ]
    is_stub_only = all(
        line in ("pass", "...", '"""', "'''") or line.startswith('"""') or line.startswith("'''")
        for line in body_lines
    ) if body_lines else True

    if not has_return and not is_stub_only:
        issues.append(
            "prompt code fence missing return statement — the function stub should "
            "include a 'return ...' line showing the expected return value/type"
        )

    return issues


def _extract_return_statement(code: str) -> str:
    """Extract the last ``return ...`` statement from *code*, handling multi-line returns.

    Returns the full (possibly multi-line) return statement, or ``""`` if none found.
    """
    lines = code.splitlines()
    return_statements: list = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("return ") or stripped == "return":
            collected = [lines[i].rstrip()]
            open_parens = stripped.count("(") - stripped.count(")")
            open_brackets = stripped.count("[") - stripped.count("]")
            open_braces = stripped.count("{") - stripped.count("}")
            ends_with_backslash = stripped.endswith("\\")
            j = i + 1
            while j < len(lines) and (
                open_parens > 0
                or open_brackets > 0
                or open_braces > 0
                or ends_with_backslash
            ):
                continuation = lines[j].rstrip()
                collected.append(continuation)
                s = continuation.strip()
                open_parens += s.count("(") - s.count(")")
                open_brackets += s.count("[") - s.count("]")
                open_braces += s.count("{") - s.count("}")
                ends_with_backslash = s.endswith("\\")
                j += 1
            return_statements.append("\n".join(collected))
            i = j
        else:
            i += 1
    return return_statements[-1] if return_statements else ""


def validate_prompt_return_vs_solution(prompt_text: str, solution_text: str) -> List[str]:
    """Compare the return statement in the prompt's code fence against the solution.

    If the prompt has a return statement that does not appear (normalized) in the
    solution code, flag it as a mismatch so the prompt can be corrected.
    Returns a list of issues (empty if OK).
    """
    issues: List[str] = []
    if not prompt_text or not solution_text:
        return issues

    # Extract code fences from the prompt
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", prompt_text, re.DOTALL)
    if not code_blocks:
        return issues

    # Find the code block containing a function definition
    func_block = ""
    for block in code_blocks:
        if re.search(r"^\s*def\s+\w+\s*\(", block, re.MULTILINE):
            func_block = block
            break
    if not func_block:
        return issues

    prompt_return = _extract_return_statement(func_block)
    if not prompt_return:
        return issues  # No return in prompt — handled by validate_prompt_function_signature

    solution_return = _extract_return_statement(solution_text)
    if not solution_return:
        return issues  # No return in solution — unusual but not a prompt mismatch issue

    # Normalize for comparison: strip each line and collapse whitespace
    def _normalize(s: str) -> str:
        return " ".join(s.split())

    prompt_norm = _normalize(prompt_return)
    solution_norm = _normalize(solution_return)

    if prompt_norm != solution_norm:
        # Also check if the prompt return text appears anywhere in the solution
        prompt_stripped = prompt_return.strip()
        if prompt_stripped not in solution_text:
            issues.append(
                f"prompt return statement does not match solution — "
                f"prompt has '{prompt_norm[:80]}' but solution has "
                f"'{solution_norm[:80]}'. Update the prompt's return line to "
                f"match the ground truth solution."
            )

    return issues


# ---------------------------------------------------------------------------
# LaTeX formatting validation
# ---------------------------------------------------------------------------

def _count_dollar_delimiters(text: str) -> Tuple[int, int]:
    """Scan *text* and return ``(single_dollar_count, double_dollar_count)``.

    A ``$$`` token counts as **one** display-math delimiter (not two
    singles).  The counts can then be checked for even-ness to detect
    mismatches.
    """
    single = 0
    double = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "$":
            if i + 1 < n and text[i + 1] == "$":
                double += 1
                i += 2
            else:
                single += 1
                i += 1
        else:
            i += 1
    return single, double


def validate_latex_formatting(text: str) -> List[str]:
    """Check for common LaTeX formatting issues in a markdown cell.

    The function strips fenced code blocks and inline code spans before
    inspecting the remaining text for:

    1. Unmatched ``$`` (inline) and ``$$`` (display) delimiters.
    2. Unmatched ``\\(`` / ``\\)`` and ``\\[`` / ``\\]`` delimiters
       (handles both single- and double-backslash escaping).
    3. Unmatched ``\\begin{env}`` / ``\\end{env}`` environments.
    4. Unbalanced curly braces ``{ }`` inside matched math regions.
    5. Common broken LaTeX commands (``\\frac``, ``\\sqrt``, ``\\text``,
       ``\\mathbb``, etc.) that are missing the required ``{…}`` argument.

    Returns a list of human-readable issue strings (empty when clean).
    """
    issues: List[str] = []
    if not text or not text.strip():
        return issues

    # --- Strip regions that should NOT be LaTeX-checked ---
    cleaned = re.sub(r"```[\w]*\n?[\s\S]*?```", "", text)  # fenced code
    cleaned = re.sub(r"`[^`\n]+`", "", cleaned)             # inline code

    # ---- 1. Dollar-sign delimiters ----
    single, double = _count_dollar_delimiters(cleaned)
    if double % 2 != 0:
        issues.append("unmatched '$$' display-math delimiter (odd count)")
    if single % 2 != 0:
        issues.append("unmatched '$' inline-math delimiter (odd count)")

    # ---- 2. Backslash-paren / backslash-bracket delimiters ----
    # Match both  \(  and  \\(  forms (1 or 2 backslashes).
    open_paren = len(re.findall(r"\\{1,2}\(", cleaned))
    close_paren = len(re.findall(r"\\{1,2}\)", cleaned))
    if open_paren != close_paren:
        issues.append(
            f"unmatched '\\(' / '\\)' inline-math delimiters "
            f"({open_paren} opening vs {close_paren} closing)"
        )

    open_bracket = len(re.findall(r"\\{1,2}\[", cleaned))
    close_bracket = len(re.findall(r"\\{1,2}\]", cleaned))
    if open_bracket != close_bracket:
        issues.append(
            f"unmatched '\\[' / '\\]' display-math delimiters "
            f"({open_bracket} opening vs {close_bracket} closing)"
        )

    # ---- 3. \begin / \end environments ----
    begins = re.findall(r"\\begin\{([^}]+)\}", cleaned)
    ends = re.findall(r"\\end\{([^}]+)\}", cleaned)
    begin_counts: Dict[str, int] = {}
    end_counts: Dict[str, int] = {}
    for env in begins:
        begin_counts[env] = begin_counts.get(env, 0) + 1
    for env in ends:
        end_counts[env] = end_counts.get(env, 0) + 1
    for env in sorted(set(begin_counts) | set(end_counts)):
        b = begin_counts.get(env, 0)
        e = end_counts.get(env, 0)
        if b != e:
            issues.append(
                f"unmatched LaTeX environment '{env}' "
                f"({b} \\begin vs {e} \\end)"
            )

    # ---- 4. Curly-brace balance inside matched math regions ----
    math_regions: List[str] = []
    # $$...$$ regions
    math_regions.extend(
        m.group(1) for m in re.finditer(r"\$\$([\s\S]*?)\$\$", cleaned)
    )
    # $...$ regions (after removing $$)
    no_display = re.sub(r"\$\$[\s\S]*?\$\$", "", cleaned)
    math_regions.extend(
        m.group(1) for m in re.finditer(r"\$([^$]+)\$", no_display)
    )
    # \(...\) regions
    math_regions.extend(
        m.group(1)
        for m in re.finditer(r"\\{1,2}\(([\s\S]*?)\\{1,2}\)", cleaned)
    )
    # \[...\] regions
    math_regions.extend(
        m.group(1)
        for m in re.finditer(r"\\{1,2}\[([\s\S]*?)\\{1,2}\]", cleaned)
    )

    brace_imbalance_reported = False
    for region in math_regions:
        depth = 0
        for ch in region:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth < 0:
                if not brace_imbalance_reported:
                    issues.append(
                        "unmatched '}' (extra closing brace) in a math expression"
                    )
                    brace_imbalance_reported = True
                break
        if depth > 0 and not brace_imbalance_reported:
            issues.append(
                "unmatched '{' (unclosed brace) in a math expression"
            )
            brace_imbalance_reported = True

    # ---- 5. Common broken LaTeX commands (missing required {…}) ----
    # These commands MUST be followed by '{' (possibly with whitespace).
    _COMMANDS_NEEDING_BRACE = [
        "frac", "sqrt", "text", "textbf", "textit", "mathrm", "mathbf",
        "mathbb", "mathcal", "hat", "bar", "tilde", "vec", "overline",
        "underline", "overbrace", "underbrace",
    ]
    for cmd in _COMMANDS_NEEDING_BRACE:
        # Find  \cmd  NOT followed by  {  or  [  (sqrt allows optional [])
        pattern = rf"\\{cmd}(?![{{a-zA-Z\[])"
        for m in re.finditer(pattern, cleaned):
            pos = m.start()
            # Grab surrounding context for the message
            snippet = cleaned[max(0, pos - 5) : pos + len(cmd) + 8].strip()
            issues.append(
                f"LaTeX command '\\{cmd}' is not followed by "
                f"'{{…}}' near: \"{snippet}\""
            )
            break  # one report per command is enough

    return issues


def validate_title_content(text: str, cell_type: str) -> List[str]:
    """Validate the cell immediately after '# Title'.

    The title cell should be a single short line of plain text — no URLs,
    code, markdown formatting, LaTeX blocks, or multi-paragraph content.
    Returns a list of specific issues found (empty if the title is clean).
    """
    issues: List[str] = []
    if not text.strip():
        issues.append("title cell is empty")
        return issues

    if cell_type == "code":
        issues.append("title is in a code cell instead of a markdown cell")
        return issues

    # URLs / links
    if re.search(r"https?://", text):
        issues.append("contains URL(s)")
    if re.search(r"\[.*?\]\(.*?\)", text):
        issues.append("contains markdown link(s)")

    # Code indicators
    if "```" in text:
        issues.append("contains code block(s)")
    if re.search(r"^\s*(?:def |class |import |from .+ import |print\()", text, re.MULTILINE):
        issues.append("contains code-like statements")

    # LaTeX block equations
    if "$$" in text:
        issues.append("contains LaTeX block equation(s)")

    # Markdown structural elements (headers, bullet lists, numbered lists)
    if re.search(r"^#{1,6}\s", text, re.MULTILINE):
        issues.append("contains markdown heading(s)")
    if re.search(r"^\s*[-*+]\s", text, re.MULTILINE):
        issues.append("contains bullet list(s)")
    if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
        issues.append("contains numbered list(s)")

    # Excessive length: a title should be a single concise line
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) > 3:
        issues.append(f"too many lines ({len(lines)}); expected a single title line")
    elif len(text.strip()) > 300:
        issues.append(f"title text too long ({len(text.strip())} chars)")

    return issues


# ---------------------------------------------------------------------------
# Common module aliases — maps alias → import statement that provides it
# ---------------------------------------------------------------------------
COMMON_MODULE_ALIASES: Dict[str, str] = {
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib.pyplot",
    "sns": "seaborn",
    "tf": "tensorflow",
    "torch": "torch",
    "sp": "scipy",
    "nx": "networkx",
    "cv2": "cv2",
    "sk": "sklearn",
    "jax": "jax",
    "jnp": "jax.numpy",
    "optax": "optax",
    "flax": "flax",
    "scipy": "scipy",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "sklearn",
    "sympy": "sympy",
    "networkx": "networkx",
    "xarray": "xarray",
    "xr": "xarray",
}


def detect_missing_imports(solution: str, test_template: str) -> List[str]:
    """Detect modules used in code but never imported.

    Combines solution and test template code and checks for common module
    aliases/names that appear as identifiers but have no corresponding
    import statement anywhere in the combined code.

    Returns a list of issue strings (empty if no problems found).
    """
    issues: List[str] = []
    if not solution and not test_template:
        return issues

    combined = (solution or "") + "\n" + (test_template or "")

    # Gather all imported names (both `import X` and `from X import ...`)
    imported_names: Set[str] = set()
    for line in combined.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            # import numpy as np  →  add both 'numpy' and 'np'
            chunk = stripped[len("import "):]
            for part in chunk.split(","):
                part = part.strip()
                if " as " in part:
                    mod, alias = part.split(" as ", 1)
                    imported_names.add(mod.strip().split(".")[0])
                    imported_names.add(alias.strip())
                else:
                    imported_names.add(part.split(".")[0].strip())
        elif stripped.startswith("from "):
            # from numpy import array  →  add 'numpy', 'array'
            chunk = stripped[len("from "):]
            parts = chunk.split("import", 1)
            if len(parts) == 2:
                mod = parts[0].strip()
                imported_names.add(mod.split(".")[0])
                for name in parts[1].split(","):
                    name = name.strip()
                    if " as " in name:
                        _, alias = name.split(" as ", 1)
                        imported_names.add(alias.strip())
                    else:
                        imported_names.add(name.strip())

    # Check for common module aliases used but not imported
    missing: List[str] = []
    for alias, module in COMMON_MODULE_ALIASES.items():
        if alias in imported_names:
            continue
        # Check if alias is used as an identifier (e.g., np.array, torch.tensor)
        pattern = rf"\b{re.escape(alias)}\s*\."
        if re.search(pattern, combined):
            missing.append(f"'{alias}' (from {module})")

    if missing:
        issues.append(
            f"Code uses {', '.join(missing)} but no corresponding import "
            f"statement was found. Add the missing import(s) to the solution "
            f"or test code."
        )

    return issues


def extract_imports(code: str) -> List[str]:
    modules: List[str] = []
    if not code:
        return modules
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            chunk = stripped[len("import ") :]
            for part in chunk.split(","):
                name = part.strip().split(" ")[0]
                if name:
                    modules.append(name.split(".")[0])
        elif stripped.startswith("from "):
            chunk = stripped[len("from ") :]
            mod = chunk.split("import")[0].strip()
            if mod:
                modules.append(mod.split(".")[0])
    return sorted(set(modules))


# ---------------------------------------------------------------------------
# Mapping from Python *import* names to their corresponding *pip* package
# names.  Only entries where the two differ need to appear here.
# ---------------------------------------------------------------------------
IMPORT_TO_PIP_MAP: Dict[str, str] = {
    "pywt": "PyWavelets",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "attr": "attrs",
    "gi": "PyGObject",
    "Crypto": "pycryptodome",
    "lxml": "lxml",
    "wx": "wxPython",
    "serial": "pyserial",
    "usb": "pyusb",
    "skimage": "scikit-image",
    "tables": "tables",
    "Bio": "biopython",
    "rdkit": "rdkit",
    "ase": "ase",
    "qiskit": "qiskit",
    "GPy": "GPy",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "cupy": "cupy",
    "faiss": "faiss-cpu",
    "magic": "python-magic",
    "dotenv": "python-dotenv",
    "jose": "python-jose",
    "jwt": "PyJWT",
    "dateutil": "python-dateutil",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "fitz": "PyMuPDF",
    "sparse": "sparse",
}


# Lock to serialise pip install calls across threads so that concurrent
# section workers don't race on the same package installation.
_pip_install_lock = threading.Lock()


def ensure_python_packages(modules: List[str]) -> List[str]:
    """Try to import each module; auto-install via pip if missing.

    Uses ``IMPORT_TO_PIP_MAP`` to translate import names to the correct
    pip package name when they differ (e.g. ``pywt`` -> ``PyWavelets``).

    Thread-safe: pip installs are serialised via ``_pip_install_lock``
    so parallel section workers never race on the same package.
    """
    if not modules:
        return []
    failed: List[str] = []
    for mod in modules:
        try:
            __import__(mod)
        except Exception:
            pip_name = IMPORT_TO_PIP_MAP.get(mod, mod)
            with _pip_install_lock:
                # Re-check after acquiring the lock — another thread may
                # have installed the package while we were waiting.
                try:
                    __import__(mod)
                    continue  # installed by another thread
                except Exception:
                    pass
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", pip_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    __import__(mod)
                except Exception:
                    failed.append(mod)
    return failed


def parse_notebook(
    input_path: Path,
) -> Tuple[Dict[str, Dict[str, str]], List[str], bool, bool]:
    with input_path.open("r", encoding="utf-8") as input_file:
        notebook = json.load(input_file)

    cells = notebook.get("cells", [])
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None
    sections: Dict[str, Dict[str, str]] = {}
    issues: List[str] = []
    metadata_found = False
    title_found = False
    expect_title_content = False  # True right after we see the # Title heading
    allowed_code_subsections = {"testing template", "solution"}

    for cell_idx, cell in enumerate(cells, start=1):
        cell_type = cell.get("cell_type")

        # --- Check the cell immediately after '# Title' ---
        if expect_title_content:
            expect_title_content = False
            if cell_type == "markdown":
                title_text = join_markdown(cell.get("source", []))
            elif cell_type == "code":
                title_text = join_code(cell.get("source", []))
            else:
                title_text = ""
            title_issues = validate_title_content(title_text, cell_type or "")
            if title_issues:
                issues.append(
                    f"Cell #{cell_idx}: Title content issue — "
                    f"{'; '.join(title_issues)}. "
                    f"The cell after '# Title' should contain only a single, "
                    f"concise plain-text title."
                )

        if cell_type == "markdown":
            text = join_markdown(cell.get("source", []))
            if text.startswith("# "):
                heading = normalize_heading(text[2:])
                current_subsection = None
                if heading.startswith("subproblem"):
                    current_section = heading
                    sections.setdefault(current_section, {})
                elif heading == "main problem":
                    current_section = "main problem"
                    sections.setdefault(current_section, {})
                elif heading == "metadata":
                    metadata_found = True
                    current_section = "metadata"
                elif heading == "title":
                    title_found = True
                    current_section = "title"
                    expect_title_content = True
                else:
                    current_section = None
            elif text.startswith("## "):
                heading = normalize_heading(text[3:])
                current_subsection = heading
                if current_section:
                    sections.setdefault(current_section, {})
                    sections[current_section].setdefault(current_subsection, "")
            else:
                if current_section and current_subsection:
                    existing = sections[current_section].get(current_subsection, "")
                    if existing:
                        sections[current_section][current_subsection] = (
                            existing + "\n\n" + text
                        )
                    else:
                        sections[current_section][current_subsection] = text
        elif cell_type == "code":
            code_text = join_code(cell.get("source", []))
            if not code_text.strip():
                continue
            code_preview = code_text[:80].replace("\n", " ").strip()
            if current_section and current_subsection in allowed_code_subsections:
                existing = sections[current_section].get(current_subsection, "")
                if existing:
                    sections[current_section][current_subsection] = (
                        existing + "\n\n" + code_text
                    )
                else:
                    sections[current_section][current_subsection] = code_text
            else:
                if current_section and current_subsection:
                    location = f"section '{current_section}', subsection '{current_subsection}'"
                elif current_section:
                    location = f"section '{current_section}' (no subsection heading found above it)"
                else:
                    location = "outside any recognized section (no '# Main Problem' or '# Subproblem N' heading found above it)"
                issues.append(
                    f"Cell #{cell_idx} (code): Code cell found outside the allowed subsections. "
                    f"Located in {location}. "
                    f"Code cells are only allowed under '## Testing Template' "
                    f"or '## Solution'. "
                    f"Code preview: \"{code_preview}...\""
                )
            if re.search(r"\b(?:pip|apt-get)\s+install\b", code_text):
                install_match = re.search(r"(?:pip|apt-get)\s+install\s+\S+", code_text)
                install_cmd = install_match.group(0) if install_match else "pip/apt-get install"
                issues.append(
                    f"Cell #{cell_idx} (code): Inline package installation detected ('{install_cmd}'). "
                    f"Dependencies should be declared in the Metadata section or a requirements file, "
                    f"not installed directly in code cells."
                )

    return sections, issues, metadata_found, title_found


def parse_scicode_json(
    input_path: Path,
) -> Tuple[
    List[Tuple[str, Dict[str, Dict[str, str]]]],
    List[str],
    List[str],
]:
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, list) or not payload:
        raise ValueError(f"JSON file does not contain a task list: {input_path}")

    parsed_tasks: List[Tuple[str, Dict[str, Dict[str, str]]]] = []
    prompt_messages: List[str] = []
    prompt_seen: Dict[str, str] = {}
    dependency_messages: List[str] = []

    for idx, entry in enumerate(payload):
        task_data = entry.get("task_data", {})
        task_label = entry.get("file_id") or task_data.get("problem_id") or str(idx + 1)
        sections: Dict[str, Dict[str, str]] = {}

        main_prompt = task_data.get("problem_description_main", "")
        if main_prompt:
            normalized = " ".join(main_prompt.split()).strip().lower()
            if normalized in prompt_seen:
                prompt_messages.append(
                    f"Duplicate prompt between '{prompt_seen[normalized]}' and '{task_label}'."
                )
            else:
                prompt_seen[normalized] = task_label

        required_deps = task_data.get("required_dependencies", "")
        if required_deps:
            dependency_messages.append(
                f"{task_label}: required dependencies listed in JSON; ensure environment matches."
            )

        main_section = {
            "prompt": task_data.get("problem_statement", ""),
            "testing template": "\n\n".join(task_data.get("general_tests", [])),
            "solution": task_data.get("general_solution", ""),
        }
        sections["main problem"] = main_section

        for step in task_data.get("sub_steps", []):
            step_number = step.get("step_number", "")
            section_key = f"subproblem {step_number}"
            sections[section_key] = {
                "prompt": step.get("problem_statement_step", ""),
                "testing template": "\n\n".join(step.get("test_cases", [])),
                "solution": step.get("ground_truth_code", ""),
            }

        parsed_tasks.append((task_label, sections))

    return parsed_tasks, prompt_messages, dependency_messages


def validate_section_keys(
    section: str,
    data: Dict[str, str],
    required_keys: List[str],
) -> List[str]:
    missing = []
    required_list = ", ".join(f"'{k.title()}'" for k in required_keys)
    for key in required_keys:
        if not data.get(key):
            missing.append(
                f"Section '{section}': Missing required subsection '## {key.title()}'. "
                f"Every problem section must contain: {required_list}."
            )
    return missing


def _extract_prompt_target_function(prompt: str, solution: str = "") -> Optional[str]:
    """Extract the primary target function name from the prompt's fenced code block.

    IMPORTANT: This function should use prompt-derived signal only for
    coverage checks. Falling back to solution can create false positives
    (e.g., "tests never call X") when prompt parsing is imperfect.
    """
    if not prompt:
        return None

    # 1) Closed fenced blocks (any language tag)
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", prompt, re.DOTALL)
    for block in code_blocks:
        funcs = extract_functions(block, top_level_only=True)
        if funcs:
            return funcs[0]

    # 2) Unclosed python fence fallback (common authoring mistake)
    m = re.search(r"```\s*python\s*\n(.*)", prompt, re.DOTALL | re.IGNORECASE)
    if m:
        funcs = extract_functions(m.group(1), top_level_only=True)
        if funcs:
            return funcs[0]

    # 3) Last-resort: direct def in prompt text (outside fenced blocks)
    funcs = extract_functions(prompt, top_level_only=False)
    if funcs:
        return funcs[0]

    return None


def _get_top_level_function_line_span(
    source: str,
    func_name: str,
) -> Optional[Tuple[int, int]]:
    """Return (start_lineno, end_lineno) for a top-level function if found."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if isinstance(start, int) and isinstance(end, int):
                return (start, end)
            # Conservative fallback when end_lineno is unavailable
            if isinstance(start, int):
                return (start, start)
    return None


def _count_function_calls_in_tests(test_code: str, func_name: str) -> int:
    """Count how many times *func_name* is invoked in *test_code* (AST-based)."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return len(re.findall(rf"\b{re.escape(func_name)}\s*\(", test_code))

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                count += 1
            elif isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                count += 1
    return count


def _analyze_assertion_patterns(test_code: str) -> Dict[str, int]:
    """Categorise assertion styles used in *test_code*.

    Returns a dict mapping pattern names to occurrence counts (only
    patterns that appear at least once are included).
    """
    patterns = {
        "equality": len(re.findall(r"\bassert\b.*==", test_code)),
        "inequality": len(re.findall(r"\bassert\b.*!=", test_code)),
        "comparison": len(re.findall(r"\bassert\b.*[<>]=?", test_code)),
        "membership": len(re.findall(r"\bassert\b.*\bin\b", test_code)),
        "identity": len(re.findall(r"\bassert\b.*\bis\b", test_code)),
        "isinstance": len(re.findall(r"\bassert\b.*isinstance", test_code)),
        "exception": len(
            re.findall(r"(?:pytest\.raises|assertRaises|with\s+.*raises)", test_code)
        ),
        "approx": len(
            re.findall(
                r"(?:approx|isclose|allclose|assert_allclose"
                r"|assert_array_almost_equal|abs\(.*?\)\s*<)",
                test_code,
            )
        ),
    }
    return {k: v for k, v in patterns.items() if v > 0}


def _detect_edge_case_patterns(test_code: str) -> List[str]:
    """Detect which common edge-case patterns are present in *test_code*."""
    found: List[str] = []
    if re.search(r"\bNone\b", test_code):
        found.append("None/null values")
    if re.search(r'\[\s*\]|""|\'\'|set\(\)|dict\(\)|tuple\(\)', test_code):
        found.append("empty collections/strings")
    if re.search(r"(?<!\w)-\d+", test_code):
        found.append("negative values")
    if re.search(r"\b0\b(?!\.\d)", test_code):
        found.append("zero values")
    if re.search(
        r"(?:raise|error|exception|invalid|illegal)", test_code, re.IGNORECASE
    ):
        found.append("error/exception handling")
    if re.search(r"(?:large|big|huge|max|overflow|\d{6,})", test_code, re.IGNORECASE):
        found.append("large/boundary values")
    if re.search(r"(?:small|min|tiny|underflow)", test_code, re.IGNORECASE):
        found.append("small/minimum values")
    return found


# -- Prompt constraint extraction & matching ---------------------------------

# Each entry maps a constraint keyword in the prompt to a regex that would
# indicate the test template addresses that constraint.
_CONSTRAINT_PATTERNS: Dict[str, Tuple[str, str, bool]] = {
    # (prompt_regex, test_regex, case_sensitive_for_test)
    "error_handling": (
        r"\b(?:raise[sd]?|throws?|errors?|exceptions?|invalid)\b",
        r"(?:raise|error|exception|invalid|pytest\.raises|assertRaises)",
        False,
    ),
    "none_handling": (
        r"\b(?:None|null|optional)\b",
        r"\bNone\b",
        True,
    ),
    "empty_input": (
        r"\bempty\b",
        r'(?:\[\s*\]|""|\'\'|set\(\)|dict\(\)|tuple\(\)|\.empty)',
        True,
    ),
    "negative_values": (
        r"\bnegative\b",
        r"(?:-\d+|negative)",
        False,
    ),
    "zero_values": (
        r"\bzero\b",
        r"(?:\b0\b|\b0\.0\b)",
        True,
    ),
    "boundary": (
        r"\b(?:boundary|edge\s*case|corner\s*case)\b",
        r"(?:max|min|boundary|edge|limit)",
        False,
    ),
    "return_type": (
        r"returns?\s+.*?(?:list|dict|tuple|array|matrix|float|int|str|bool)",
        r"(?:isinstance|type\()",
        True,
    ),
    "value_range": (
        r"\b(?:between|range|at\s*least|at\s*most|minimum|maximum|greater|less|positive)\b",
        r"(?:[<>]=?|between|range|abs\()",
        False,
    ),
}


def _extract_prompt_constraints(prompt: str) -> List[Tuple[str, str]]:
    """Extract testable constraints from the prompt text.

    Returns a list of ``(constraint_type, human_description)`` tuples for
    each constraint keyword detected in the prompt.
    """
    if not prompt:
        return []

    prompt_lower = prompt.lower()
    labels = {
        "error_handling": "error/exception handling",
        "none_handling": "None/null value handling",
        "empty_input": "empty input handling",
        "negative_values": "negative value handling",
        "zero_values": "zero value handling",
        "boundary": "boundary/edge case handling",
        "return_type": "return type verification",
        "value_range": "value range/constraint verification",
    }
    constraints: List[Tuple[str, str]] = []
    for ctype, (prompt_re, _test_re, _cs) in _CONSTRAINT_PATTERNS.items():
        if re.search(prompt_re, prompt_lower if not _cs else prompt):
            constraints.append((ctype, labels[ctype]))
    return constraints


def _check_uncovered_constraints(
    constraints: List[Tuple[str, str]],
    test_code: str,
) -> List[str]:
    """Return human descriptions of constraints NOT addressed in *test_code*."""
    if not constraints or not test_code:
        return [desc for _, desc in constraints] if constraints else []

    test_lower = test_code.lower()
    uncovered: List[str] = []
    for ctype, cdesc in constraints:
        _prompt_re, test_re, case_sensitive = _CONSTRAINT_PATTERNS[ctype]
        text = test_code if case_sensitive else test_lower
        if not re.search(test_re, text):
            uncovered.append(cdesc)
    return uncovered


def static_test_coverage_analysis(
    section: str,
    prompt: str,
    test_template: str,
    solution: str,
) -> Tuple[List[str], List[str]]:
    """Statically analyse how well the testing template covers the prompt's
    requirements.

    Checks performed (all AST / regex based, no runtime tracing):
      1. Target function calls — does the test invoke the main function?
      2. Assertion depth — are there enough assertions?
      3. Assertion variety — equality, comparison, membership, etc.
      4. Edge-case patterns — None, empty, negative, zero, boundary, etc.
      5. Prompt-constraint coverage — do tests address constraints that the
         prompt explicitly mentions (e.g. error handling, value ranges)?

    Returns ``(issues, results)`` consistent with the rest of the validator.
    """
    issues: List[str] = []
    results: List[str] = []

    if not test_template or not test_template.strip():
        return issues, results

    # --- 1. Identify the target function from the prompt ---
    target_func = _extract_prompt_target_function(prompt, solution)

    # --- 2. Test-function & call counts ---
    test_funcs = extract_test_functions(test_template)
    call_count = 0
    call_signal_ok = True
    if target_func:
        call_count = _count_function_calls_in_tests(test_template, target_func)
        if call_count == 0:
            call_signal_ok = False
            issues.append(
                f"Section '{section}': Tests never call the target function "
                f"'{target_func}()' defined in the prompt. Tests should invoke "
                f"the function with various inputs to verify correctness."
            )
        else:
            min_expected_calls = max(2, len(test_funcs)) if test_funcs else 2
            if call_count < min_expected_calls:
                call_signal_ok = False
                issues.append(
                    f"Section '{section}': Target function '{target_func}()' is "
                    f"called only {call_count} time(s) across {len(test_funcs)} "
                    f"test function(s). Add more distinct call scenarios to "
                    f"improve behavioral coverage."
                )
        if call_count < 2 and call_signal_ok:
            issues.append(
                f"Section '{section}': Target function '{target_func}()' is "
                f"called only {call_count} time(s) in tests. Consider adding "
                f"more test scenarios with different inputs."
            )

    # --- 3. Assertion analysis ---
    assertion_count = count_assertions(test_template)
    assertion_types = _analyze_assertion_patterns(test_template)
    assertion_signal_ok = True
    variety_signal_ok = True

    min_assertions = max(3, len(test_funcs)) if test_funcs else 3
    if assertion_count < min_assertions:
        assertion_signal_ok = False
        issues.append(
            f"Section '{section}': Insufficient assertion depth — found "
            f"{assertion_count} assertion(s) across {len(test_funcs)} test "
            f"function(s). Add stronger expected-output checks."
        )

    if assertion_count > 0 and len(assertion_types) == 1:
        variety_signal_ok = False
        only_type = list(assertion_types.keys())[0]
        issues.append(
            f"Section '{section}': Tests use only one assertion pattern "
            f"('{only_type}'). Consider adding variety (e.g. type checks, "
            f"approximate comparisons, edge-case checks)."
        )

    # --- 4. Edge-case patterns ---
    edge_cases = _detect_edge_case_patterns(test_template)
    edge_signal_ok = True
    if len(edge_cases) == 0:
        edge_signal_ok = False
        issues.append(
            f"Section '{section}': Edge-case coverage appears weak — no common "
            f"edge-patterns (e.g. boundary/empty/None/invalid inputs) were "
            f"detected in tests."
        )

    # --- 5. Prompt-constraint coverage ---
    prompt_constraints = _extract_prompt_constraints(prompt)
    uncovered_constraints = _check_uncovered_constraints(
        prompt_constraints, test_template
    )
    constraints_signal_ok = True

    if uncovered_constraints:
        constraints_signal_ok = False
        constraints_str = ", ".join(uncovered_constraints)
        issues.append(
            f"Section '{section}': The prompt mentions constraints that tests "
            f"may not cover: {constraints_str}. Consider adding test cases "
            f"for these requirements."
        )

    # --- 6. Composite test-quality score ---
    quality_checks: List[bool] = []
    if target_func:
        quality_checks.append(call_signal_ok)
    quality_checks.extend([assertion_signal_ok, variety_signal_ok, edge_signal_ok])
    if prompt_constraints:
        quality_checks.append(constraints_signal_ok)
    if quality_checks:
        quality_score = round(100.0 * sum(1 for ok in quality_checks if ok) / len(quality_checks))
        if quality_score < 60:
            issues.append(
                f"Section '{section}': Low test quality score ({quality_score}%) "
                f"based on call/assertion/edge/constraint signals. Improve test "
                f"scenario diversity and assertion rigor."
            )
    else:
        quality_score = 100

    # --- Build summary result ---
    covered_count = len(prompt_constraints) - len(uncovered_constraints)
    detail_parts: List[str] = []
    if target_func:
        detail_parts.append(f"target='{target_func}()'")
    detail_parts.append(f"test_funcs={len(test_funcs)}")
    detail_parts.append(f"calls={call_count}")
    detail_parts.append(f"asserts={assertion_count}")
    detail_parts.append(f"assert_types={len(assertion_types)}")
    detail_parts.append(f"edge_patterns={len(edge_cases)}")
    detail_parts.append(f"quality_score={quality_score}%")
    if prompt_constraints:
        detail_parts.append(
            f"constraints={covered_count}/{len(prompt_constraints)}"
        )

    results.append(
        f"PASS {section}: static_coverage ({', '.join(detail_parts)})"
    )

    return issues, results


def run_section_tests(
    section: str,
    data: Dict[str, str],
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
) -> Tuple[List[str], List[str]]:
    """Run validation and tests for a single problem section.

    Each section gets its own **isolated namespace** so that the
    self-containment of each problem's solution + tests is verified
    independently.  Nothing leaks between sections.
    """
    issues: List[str] = []
    results: List[str] = []

    prompt = data.get("prompt", "")
    background = data.get("background", "")
    test_template = data.get("testing template", "")
    solution = data.get("solution", "")

    # --- Validate prompt contains function signature in code block ---
    prompt_sig_issues = validate_prompt_function_signature(prompt)
    if prompt_sig_issues:
        issues.append(
            f"Section '{section}': {'; '.join(prompt_sig_issues)}. "
            f"The '## Prompt' should include a fenced code block with the "
            f"function signature (def, docstring, return) for the model to implement."
        )

    # --- Validate prompt return statement matches solution return ---
    prompt_return_issues = validate_prompt_return_vs_solution(prompt, solution)
    if prompt_return_issues:
        issues.append(
            f"Section '{section}': {'; '.join(prompt_return_issues)}. "
            f"The '## Prompt' return line should match the return statement "
            f"in the '## Solution'."
        )

    # --- Validate LaTeX formatting in Prompt and Background ---
    for field_name, field_text in [("Prompt", prompt), ("Background", background)]:
        latex_issues = validate_latex_formatting(field_text)
        if latex_issues:
            joined = "; ".join(latex_issues)
            issues.append(
                f"Section '{section}': LaTeX formatting issue in '## {field_name}': "
                f"{joined}."
            )

    # Extract the target function(s) from the prompt's fenced code block.
    # (Notebooks do not have a separate '## Function Declaration' section.)
    prompt_code_blocks = re.findall(r"```[\w]*\n?(.*?)```", prompt, re.DOTALL)
    declared_functions: List[str] = []
    for block in prompt_code_blocks:
        declared_functions.extend(extract_functions(block, top_level_only=True))
    declared_functions = list(dict.fromkeys(declared_functions))  # dedupe
    solution_functions = extract_functions(solution, top_level_only=True)
    declared_tests = extract_test_functions(test_template)
    is_procedural = has_procedural_tests(test_template)

    # --- Detect solution/test code duplication ---
    if detect_solution_test_duplication(solution, test_template):
        issues.append(
            f"Section '{section}': solution cell contains duplicated test code from testing template."
        )

    solution_test_artifacts = detect_solution_test_artifacts(solution)
    if solution_test_artifacts:
        issues.append(
            f"Section '{section}': Solution code contains test artifacts "
            f"({', '.join(solution_test_artifacts)}). "
            f"Move tests and test-only constructs to '## Testing Template'."
        )

    template_artifacts = detect_test_template_artifacts(test_template)
    if template_artifacts:
        issues.append(
            f"Section '{section}': Testing template contains test-runner boilerplate "
            f"({', '.join(template_artifacts)}). "
            f"Define individual 'test_*()' functions instead of a manual test runner."
        )

    # --- Detect explicit calls to test functions (test_*()) and runner functions ---
    # Test functions should only be *defined*, never explicitly called.
    # The validator's test runner discovers and invokes them automatically.
    # Runner functions (run(), run_all(), run_tests(), etc.) should not exist.
    all_test_func_names = extract_test_functions(test_template) + extract_test_functions(solution)
    all_test_func_names = list(dict.fromkeys(all_test_func_names))  # dedupe, preserve order

    # Combine test_* names + known runner names for call detection
    all_flagged_names = list(dict.fromkeys(
        all_test_func_names + list(TEST_RUNNER_NAMES)
    ))

    if all_flagged_names:
        # Check solution code
        solution_calls = detect_test_function_calls(solution, all_flagged_names)
        sol_test_calls = [n for n in solution_calls if n.startswith("test_")]
        sol_runner_calls = [n for n in solution_calls if n in TEST_RUNNER_NAMES]

        if sol_test_calls:
            calls_str = ", ".join(f"'{n}()'" for n in sol_test_calls)
            issues.append(
                f"Section '{section}': Solution code explicitly calls test function(s): "
                f"{calls_str}. Remove these calls — test functions should not be "
                f"invoked in the solution. The validator runs them automatically."
            )
        if sol_runner_calls:
            calls_str = ", ".join(f"'{n}()'" for n in sol_runner_calls)
            issues.append(
                f"Section '{section}': Solution code explicitly calls runner function(s): "
                f"{calls_str}. Remove these calls — runner/executor functions should "
                f"not appear in the solution."
            )

        # Check testing template
        template_calls = detect_test_function_calls(test_template, all_flagged_names)
        tpl_test_calls = [n for n in template_calls if n.startswith("test_")]
        tpl_runner_calls = [n for n in template_calls if n in TEST_RUNNER_NAMES]

        if tpl_test_calls:
            calls_str = ", ".join(f"'{n}()'" for n in tpl_test_calls)
            issues.append(
                f"Section '{section}': Testing template explicitly calls test function(s): "
                f"{calls_str}. Remove these calls — only define test functions with "
                f"'def test_*():'. The validator discovers and runs them automatically."
            )
        if tpl_runner_calls:
            calls_str = ", ".join(f"'{n}()'" for n in tpl_runner_calls)
            issues.append(
                f"Section '{section}': Testing template explicitly calls runner function(s): "
                f"{calls_str}. Remove these calls — define individual 'test_*()' "
                f"functions instead. The validator runs them automatically."
            )

    required_imports = extract_imports(solution) + extract_imports(test_template)
    failed_imports = ensure_python_packages(sorted(set(required_imports)))
    if failed_imports:
        issues.append(
            f"Section '{section}': Failed to install required imports: "
            f"{', '.join(sorted(set(failed_imports)))}. "
            f"Install them in the venv and re-run."
        )

    # --- Check for modules used but never imported ---
    missing_import_issues = detect_missing_imports(solution, test_template)
    for mi in missing_import_issues:
        issues.append(f"Section '{section}': {mi}")

    if declared_functions:
        solution_set = set(solution_functions)

        for name in declared_functions:
            if name not in solution_set:
                issues.append(
                    f"Section '{section}': Function '{name}()' is specified in "
                    f"the '## Prompt' but is not implemented in '## Solution'. "
                    f"The solution code must define all functions from the prompt."
                )

    if declared_functions and test_template:
        # Only require the main (public) function to be referenced by tests.
        # Private helper functions (starting with '_') declared in the prompt
        # are internal implementation details and not required to be called
        # directly by test cases.
        public_declared = [f for f in declared_functions if not f.startswith("_")]

        for func_name in public_declared:
            if func_name not in test_template:
                issues.append(
                    f"Section '{section}': Test code in '## Testing Template' does not "
                    f"reference function '{func_name}()'. Tests should call and validate "
                    f"the declared function to ensure correctness."
                )

        # --- Per-test-function checks ---
        # Check each individual test_ function for:
        #   1. assert statements (a test with no assert is useless)
        #   2. reference to the main declared function (a test that only
        #      exercises a helper is not testing the required implementation)
        test_bodies = extract_test_function_bodies(test_template)
        if test_bodies:
            for func_name, func_body in test_bodies:
                if count_assertions(func_body) == 0:
                    issues.append(
                        f"Section '{section}': Test function '{func_name}()' has no "
                        f"'assert' statements. Each test function must include assert "
                        f"statements to verify expected behavior "
                        f"(e.g., assert result == expected_value)."
                    )
                # Check if this test function references the main declared function
                if public_declared:
                    refs_main = any(fn in func_body for fn in public_declared)
                    if not refs_main:
                        issues.append(
                            f"Section '{section}': Test function '{func_name}()' does not "
                            f"reference the main declared function "
                            f"({', '.join(repr(f) for f in public_declared)}). "
                            f"Each test should call and validate the main function, "
                            f"not just helper functions."
                        )
        elif count_assertions(test_template) == 0:
            # No test_ functions found at all — fall back to whole-template check
            issues.append(
                f"Section '{section}': No 'assert' statements found in '## Testing Template'. "
                f"Tests must include assert statements to verify expected behavior "
                f"(e.g., assert result == expected_value)."
            )

    # Only check test references for functions explicitly declared in the
    # prompt's code block.  Do NOT fall back to inferred solution
    # functions — helper functions defined only in the solution cell are
    # not required to be referenced by the tests.
    if declared_functions and test_template:
        # Only require public functions to be referenced, not _helpers
        public_declared = [f for f in declared_functions if not f.startswith("_")]
        missing = list(dict.fromkeys(name for name in public_declared if name not in test_template))
        if missing:
            preview = ", ".join(missing[:5])
            suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            issues.append(
                f"Section '{section}': tests do not reference declared function(s): {preview}{suffix}."
            )

    # --- Static test-coverage analysis against prompt constraints ---
    # Runs regardless of --no-run-test since it's purely code-based.
    cov_issues, cov_results = static_test_coverage_analysis(
        section, prompt, test_template, solution,
    )
    issues.extend(cov_issues)
    results.extend(cov_results)

    if not run_tests:
        if not declared_tests and not is_procedural:
            issues.append(
                f"Section '{section}': No test functions (def test_*) found in "
                f"'## Testing Template'. Add at least one function starting with 'test_' "
                f"to validate the solution."
            )
        return issues, results

    # --- Fresh isolated namespace for this section ---
    # Each problem section (subproblem / main problem) gets its own clean
    # namespace.  Solution and tests share it, but nothing leaks in from
    # other sections.  This ensures each problem is fully self-contained.
    namespace: Dict[str, object] = {}

    callable_tests: Dict[str, Callable[[], object]] = {}
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    redirect_stdout = contextlib.redirect_stdout(stdout_buffer)
    redirect_stderr = contextlib.redirect_stderr(stderr_buffer)

    if quiet:
        stdout_context = redirect_stdout
        stderr_context = redirect_stderr
    else:
        stdout_context = contextlib.nullcontext()
        stderr_context = contextlib.nullcontext()

    # --- Execute solution and test template separately to avoid masking errors ---
    # If coverage.py is available, instrument execution to measure how much
    # of the solution code the tests actually exercise.
    solution_tmp_path: Optional[str] = None
    cov: object = None  # coverage.Coverage instance (if available)

    solution_exec_failed = False
    solution_exec_error = None

    try:  # outer try/finally ensures temp-file cleanup
        # Write solution to a temp file so coverage.py can map lines
        if HAS_COVERAGE and solution:
            tmp_fd = tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", prefix="sol_", delete=False,
                encoding="utf-8",
            )
            tmp_fd.write(solution)
            tmp_fd.close()
            solution_tmp_path = tmp_fd.name

        if solution:
            try:
                with stdout_context, stderr_context:
                    if solution_tmp_path:
                        # Compile with the temp path so coverage recognises
                        # the executed lines as belonging to that file.
                        sol_code = compile(solution, solution_tmp_path, "exec")
                        # Start coverage *before* exec so top-level lines count
                        cov = coverage_lib.Coverage(
                            data_file=None,       # in-memory only
                            source=[solution_tmp_path],
                            branch=False,
                        )
                        cov.start()
                        exec(sol_code, namespace)
                    else:
                        exec(solution, namespace)
                results.append(f"PASS {section}: solution_execution")
            except Exception as exc:  # noqa: BLE001
                solution_exec_failed = True
                solution_exec_error = exc
                issues.append(
                    f"FAIL {section}: solution_execution -> "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            issues.append(
                f"FAIL {section}: solution_execution -> no solution code provided"
            )

        test_exec_failed = False
        if test_template:
            try:
                with stdout_context, stderr_context:
                    exec(test_template, namespace)
            except Exception as exc:  # noqa: BLE001
                test_exec_failed = True
                issues.append(
                    f"Section '{section}': test exec failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        # --- Structural check: report missing test_ functions ---
        test_functions = sorted(
            name
            for name, obj in namespace.items()
            if name.startswith("test_") and callable(obj)
        )

        for test_name in test_functions:
            test_obj = namespace.get(test_name)
            if callable(test_obj):
                callable_tests[test_name] = test_obj

        if not callable_tests and not is_procedural:
            issues.append(
                f"Section '{section}': No callable test functions (def test_*) "
                f"were found after executing the code. Ensure the Testing "
                f"Template defines at least one function whose name starts "
                f"with 'test_'."
            )
            if solution_exec_failed or test_exec_failed:
                return issues, results
        elif not callable_tests and is_procedural:
            if not solution_exec_failed and not test_exec_failed:
                results.append(
                    f"PASS {section}: procedural tests (inline assertions)"
                )
            return issues, results

        if solution_exec_failed or test_exec_failed:
            return issues, results

        with stdout_context, stderr_context:
            for test_name, test_func in callable_tests.items():
                try:
                    test_func()
                    results.append(f"PASS {section}: {test_name}")
                except Exception as exc:  # noqa: BLE001
                    issues.append(
                        f"FAIL {section}: {test_name}() raised "
                        f"{type(exc).__name__}: {exc}. The test did not pass "
                        f"-- review the solution logic and test expectations."
                    )

        # ----- Runtime coverage analysis (coverage.py) -----
        if cov is not None and solution_tmp_path:
            try:
                cov.stop()
                cov.save()
                # analysis2 returns:
                #   (filename, executable, excluded, missing, formatted_missing)
                _, executable, _, missing, _ = cov.analysis2(solution_tmp_path)

                executable_set = set(executable)
                missing_set = set(missing)

                # Prefer coverage over the prompt target function body when we
                # can identify it. This avoids over-penalising notebooks for
                # unrelated helper/setup lines in large solution cells.
                coverage_scope = "solution"
                target_func = _extract_prompt_target_function(prompt, solution)
                if target_func:
                    span = _get_top_level_function_line_span(solution, target_func)
                    if span:
                        start_ln, end_ln = span
                        scoped_executable = {
                            ln for ln in executable_set if start_ln <= ln <= end_ln
                        }
                        if scoped_executable:
                            executable_set = scoped_executable
                            missing_set = missing_set.intersection(scoped_executable)
                            coverage_scope = f"target function '{target_func}()'"

                total_executable = len(executable_set)
                total_missing = len(missing_set)
                if total_executable > 0:
                    covered = total_executable - total_missing
                    pct = round(covered * 100.0 / total_executable)
                    results.append(
                        f"INFO {section}: runtime_coverage[{coverage_scope}] = {pct}% "
                        f"({covered}/{total_executable} executable lines)"
                    )
                    if pct < COVERAGE_MIN_THRESHOLD:
                        results.append(
                            f"INFO {section}: runtime_coverage[{coverage_scope}] below "
                            f"threshold ({pct}% < {COVERAGE_MIN_THRESHOLD}%). "
                            f"Used for diagnostics only."
                        )
                cov = None  # prevent double-stop in finally
            except Exception:  # noqa: BLE001
                pass  # coverage analysis is best-effort

    finally:
        # Stop coverage if still running (early return paths)
        if cov is not None:
            try:
                cov.stop()
            except Exception:  # noqa: BLE001
                pass
        # Clean up temp file
        if solution_tmp_path:
            try:
                os.unlink(solution_tmp_path)
            except OSError:
                pass

    return issues, results


def validate_sections(
    sections: Dict[str, Dict[str, str]],
    required_keys: List[str],
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    results: List[str] = []

    def _section_sort_key(item: Tuple[str, Dict[str, str]]) -> Tuple[int, int, str]:
        key = item[0]
        if key == "main problem":
            return (0, 0, key)
        if key.startswith("subproblem"):
            match = re.search(r"subproblem\s+(\d+)", key)
            if match:
                return (1, int(match.group(1)), key)
            return (1, 9999, key)
        return (2, 0, key)

    sorted_sections = sorted(sections.items(), key=_section_sort_key)

    # Structural validation (fast) — always sequential
    for section, data in sorted_sections:
        issues.extend(validate_section_keys(section, data, required_keys))

    # --- Test execution: run sections sequentially ---
    # Explicitly avoid parallel section execution to keep RNG usage and
    # execution order deterministic across solution + tests for a section.
    # This mirrors typical notebook usage where the main solution is run
    # followed by its tests in the same thread.
    for section, data in sorted_sections:
        section_issues, section_results = run_section_tests(
            section,
            data,
            run_tests,
            quiet,
            assert_checks,
        )
        issues.extend(section_issues)
        results.extend(section_results)

    if not sections:
        issues.append(
            "No valid problem sections found in the notebook. "
            "The notebook must contain a '# Main Problem' heading AND at least 2 "
            "'# Subproblem N' headings, each with the required subsections "
            "(Prompt, Background, Testing Template, Solution)."
        )
    else:
        # Strictly require both main problem and at least 2 subproblems
        has_main = "main problem" in sections
        subproblem_sections = [
            k for k in sections if k.startswith("subproblem")
        ]
        if not has_main:
            issues.append(
                "Missing '# Main Problem' section. The notebook must contain "
                "a '# Main Problem' heading with the required subsections "
                "(Prompt, Background, Testing Template, Solution)."
            )
        if len(subproblem_sections) < 2:
            issues.append(
                f"Found {len(subproblem_sections)} subproblem section(s), but at least 2 are required. "
                f"The notebook must contain at least 2 '# Subproblem N' headings "
                f"(e.g. '# Subproblem 1', '# Subproblem 2') each with the required subsections "
                f"(Prompt, Background, Testing Template, Solution)."
            )

    return issues, results


def validate_notebook(
    input_path: Path,
    log_folder: Path,
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
) -> Tuple[Path, List[str], List[str]]:
    sections, parse_issues, metadata_found, title_found = parse_notebook(input_path)
    issues, results = validate_sections(
        sections,
        NOTEBOOK_SECTION_KEYS,
        run_tests,
        quiet,
        assert_checks,
    )
    issues = parse_issues + issues
    if not metadata_found:
        issues.append(
            "Missing '# Metadata' section. The notebook must include a top-level "
            "'# Metadata' heading containing problem metadata (field, subfield, difficulty, etc.)."
        )
    if not title_found:
        issues.append(
            "Missing '# Title' section. The notebook must include a top-level "
            "'# Title' heading with a clear, descriptive title for the scientific problem."
        )

    # --- Print per-section test summary to console ---
    _print_section_test_summary(input_path.name, sections, issues, results)

    return issues, results


def _print_section_test_summary(
    notebook_name: str,
    sections: Dict[str, Dict[str, str]],
    issues: List[str],
    results: List[str],
) -> None:
    """Print a clear pass/fail summary for each section (subfunction / main function)."""
    print(f"\n{'='*70}")
    print(f"  TEST RESULTS: {notebook_name}")
    print(f"{'='*70}")

    section_names = sorted(sections.keys())
    if not section_names:
        print("  No sections found.")
        print(f"{'='*70}\n")
        return

    total_pass = 0
    total_fail = 0

    for section in section_names:
        # Determine display label
        if section == "main problem":
            label = "Main Function"
        elif section.startswith("subproblem"):
            label = f"Sub-Function ({section})"
        else:
            label = section

        # Collect PASS/FAIL entries belonging to this section
        section_passes = [r for r in results if r.startswith(f"PASS {section}:")]
        section_fails = [i for i in issues if i.startswith(f"FAIL {section}:")]
        section_issues = [
            i for i in issues
            if section in i and not i.startswith("FAIL ") and not i.startswith("PASS ")
        ]

        pass_count = len(section_passes)
        fail_count = len(section_fails)
        total_pass += pass_count
        total_fail += fail_count

        if fail_count == 0 and pass_count > 0:
            status_icon = "PASS"
        elif fail_count > 0:
            status_icon = "FAIL"
        else:
            status_icon = "----"

        print(f"\n  [{status_icon}] {label}")
        for entry in section_passes:
            test_name = entry.split(":", 1)[1].strip() if ":" in entry else entry
            print(f"         PASS  {test_name}")
        for entry in section_fails:
            test_name = entry.split(":", 1)[1].strip() if ":" in entry else entry
            print(f"         FAIL  {test_name}")
        for entry in section_issues:
            print(f"         WARN  {entry}")

    print(f"\n{'-'*70}")
    overall = "ALL TESTS PASSED" if total_fail == 0 and total_pass > 0 else (
        "SOME TESTS FAILED" if total_fail > 0 else "NO TESTS EXECUTED"
    )
    print(f"  Summary: {total_pass} passed, {total_fail} failed  =>  {overall}")
    print(f"{'='*70}\n")


def validate_scicode_json(
    input_path: Path,
    log_folder: Path,
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
) -> Tuple[Path, Path]:
    tasks, prompt_messages, dependency_messages = parse_scicode_json(input_path)

    all_results: List[str] = []
    all_issues: List[str] = []
    per_task: Dict[str, Tuple[List[str], List[str]]] = {}

    for task_label, sections in tasks:
        issues, results = validate_sections(
            sections,
            JSON_SECTION_REQUIRED_KEYS,
            run_tests,
            quiet,
            assert_checks,
        )
        per_task[task_label] = (issues, results)
        all_results.extend(f"{task_label}: {result}" for result in results)
        all_issues.extend(f"{task_label}: {issue}" for issue in issues)

    any_failures = any(
        collect_test_failures(issues, results)
        for issues, results in per_task.values()
    )
    status = "SUCCESS" if not all_issues and not any_failures else "FAILED"
    timestamp = datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
    log_folder.mkdir(parents=True, exist_ok=True)
    log_path = log_folder / f"test_report_{input_path.stem}_{timestamp}.log"

    lines = [
        f"Status: {status}",
        f"SciCode JSON: {input_path}",
        f"Timestamp: {timestamp}",
        "",
        "Results:",
    ]
    if all_results:
        lines.extend(f"- {result}" for result in all_results)
    else:
        lines.append("- No tests executed.")

    if prompt_messages:
        lines.append("")
        lines.append("Duplicate Prompts:")
        lines.extend(f"- {message}" for message in prompt_messages)

    if dependency_messages:
        lines.append("")
        lines.append("Dependency Checks:")
        lines.extend(f"- {message}" for message in dependency_messages)

    if all_issues:
        lines.append("")
        lines.append("Issues:")
        lines.extend(f"- {issue}" for issue in all_issues)

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_folder = Path(SUMMARY_FOLDER_DEFAULT)
    summary_folder.mkdir(parents=True, exist_ok=True)
    summary_path = summary_folder / f"summary_{input_path.stem}_{timestamp}.csv"

    rows = []
    for task_label, (issues, results) in sorted(per_task.items()):
        fail_count = len(collect_test_failures(issues, results))
        status = determine_status(issues, results)
        rows.append(
            {
                "Task": task_label,
                "Status": status,
                "IssueCount": len(issues),
                "FailCount": fail_count,
                "IssueDetails": "; ".join(issues),
                "LogFile": str(log_path),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Task",
                "Status",
                "IssueCount",
                "FailCount",
                "IssueDetails",
                "LogFile",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return log_path, summary_path


# ---------------------------------------------------------------------------
# Parallel worker — isolated subprocess per notebook
# ---------------------------------------------------------------------------
# Uses multiprocessing.Process (one per notebook) managed by a
# ThreadPoolExecutor for concurrency control.  Each notebook runs in a
# **fully isolated** process so that a crash (OOM, segfault, etc.) in one
# notebook does NOT poison the remaining notebooks — unlike
# ProcessPoolExecutor where a single abrupt worker death breaks the
# entire pool (BrokenProcessPool).
# ---------------------------------------------------------------------------

def _validate_notebook_worker(
    notebook: Path,
    log_folder: Path,
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
) -> Tuple[Path, List[str], List[str]]:
    """Run validate_notebook in a worker process.

    Returns (notebook_path, issues, results).
    """
    try:
        issues, results = validate_notebook(
            notebook, log_folder, run_tests, quiet, assert_checks,
        )
        return notebook, issues, results
    except Exception as exc:
        return notebook, [f"Worker error: {type(exc).__name__}: {exc}"], []


def _mp_worker_target(
    result_queue: multiprocessing.Queue,
    notebook: Path,
    log_folder: Path,
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
) -> None:
    """Target function for multiprocessing.Process — validates one notebook
    and puts the result on a Queue."""
    try:
        issues, results = validate_notebook(
            notebook, log_folder, run_tests, quiet, assert_checks,
        )
        result_queue.put((str(notebook), issues, results))
    except Exception as exc:
        result_queue.put(
            (str(notebook), [f"Worker error: {type(exc).__name__}: {exc}"], [])
        )


def _validate_notebook_isolated(
    notebook: Path,
    log_folder: Path,
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
    timeout: int,
) -> Tuple[Path, List[str], List[str]]:
    """Run a single notebook validation in a fully isolated subprocess.

    Spawns a dedicated ``multiprocessing.Process`` for the notebook.
    If the process crashes or exceeds *timeout* seconds, only this one
    notebook is affected — no cascade failure.
    """
    result_queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_mp_worker_target,
        args=(result_queue, notebook, log_folder, run_tests, quiet, assert_checks),
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        # Timed out — kill and report
        proc.kill()
        proc.join(5)
        return notebook, [f"Worker timeout: exceeded {timeout}s"], []

    if proc.exitcode is not None and proc.exitcode != 0:
        # Process crashed (segfault, OOM-killed, etc.)
        return notebook, [
            f"Worker error: process exited with code {proc.exitcode} "
            f"(likely OOM or segfault)"
        ], []

    # Retrieve result from queue
    try:
        _, issues, results = result_queue.get_nowait()
        return notebook, issues, results
    except Exception:
        return notebook, ["Worker error: no result returned from subprocess"], []


def _parallel_validate_notebooks(
    notebooks: List[Path],
    log_folder: Path,
    run_tests: bool,
    quiet: bool,
    assert_checks: bool,
    num_workers: int,
    timeout: int,
    checkpoint_path: Optional[Path],
    checkpoint_every: int,
    root: Optional[Path],
    per_notebook: Optional[Dict[Path, Tuple[List[str], List[str]]]] = None,
) -> Dict[Path, Tuple[List[str], List[str]]]:
    """Validate a list of notebooks using parallel worker processes.

    Each notebook is validated in a **fully isolated** subprocess
    (``multiprocessing.Process``).  A ``ThreadPoolExecutor`` limits
    concurrency to *num_workers* notebooks at a time.  If any single
    subprocess crashes, only that notebook is marked as failed — the
    remaining notebooks continue unaffected.
    """
    per_notebook = per_notebook or {}
    progress = ProgressTracker(len(notebooks))
    checkpoint_every = max(1, int(checkpoint_every))
    since_checkpoint = 0

    if num_workers <= 1 or not run_tests:
        # Sequential (fast for structure-only, or single worker)
        for notebook in notebooks:
            issues, results = validate_notebook(
                notebook, log_folder, run_tests, quiet, assert_checks,
            )
            per_notebook[notebook] = (issues, results)
            progress.update()
            since_checkpoint += 1
            if checkpoint_path and since_checkpoint >= checkpoint_every:
                save_checkpoint(per_notebook, checkpoint_path, root)
                since_checkpoint = 0
    else:
        # Parallel — isolated subprocess per notebook, concurrency via threads
        futures = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for notebook in notebooks:
                future = executor.submit(
                    _validate_notebook_isolated,
                    notebook, log_folder, run_tests, quiet, assert_checks,
                    timeout,
                )
                futures[future] = notebook

            for future in as_completed(futures):
                nb = futures[future]
                try:
                    _, issues, results = future.result()
                    per_notebook[nb] = (issues, results)
                except Exception as exc:
                    per_notebook[nb] = (
                        [f"Worker error: {type(exc).__name__}: {exc}"],
                        [],
                    )
                progress.update()
                since_checkpoint += 1
                if checkpoint_path and since_checkpoint >= checkpoint_every:
                    save_checkpoint(per_notebook, checkpoint_path, root)
                    since_checkpoint = 0

    progress.finish()
    if checkpoint_path:
        save_checkpoint(per_notebook, checkpoint_path, root)
    return per_notebook


def main() -> int:
    args = parse_args()
    run_tests = not args.no_run_test
    
    # Generate run ID for better log organization
    run_timestamp = datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
    args.log_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_path

    resume_data: Dict[str, Tuple[List[str], List[str]]] = {}
    resume_worker_error: Set[str] = set()
    if args.resume:
        if not args.resume.exists():
            print(f"ERROR: Resume file not found: {args.resume}", file=sys.stderr)
            return 2
        print(f"Resuming from checkpoint: {args.resume}", file=sys.stderr)
        resume_data, resume_worker_error = load_checkpoint(args.resume)
        if checkpoint_path is None:
            checkpoint_path = args.resume

    if checkpoint_path is None:
        checkpoint_path = args.log_folder / f"{run_timestamp}_checkpoint.json"

    def _resolve_resume(
        notebooks: List[Path],
        root: Optional[Path],
    ) -> Tuple[List[Path], Dict[Path, Tuple[List[str], List[str]]]]:
        if not resume_data:
            return notebooks, {}
        name_map = {p.name: p for p in notebooks}
        prior: Dict[Path, Tuple[List[str], List[str]]] = {}
        skip_keys: Set[str] = set()
        for key, val in resume_data.items():
            if key in resume_worker_error:
                continue
            skip_keys.add(key)
            cand = (root / key) if root is not None else None
            if cand is not None and cand.exists():
                prior[cand] = val
                continue
            # fallback: match by filename
            by_name = name_map.get(Path(key).name)
            if by_name is not None:
                prior[by_name] = val
        remaining: List[Path] = []
        skip_names = {Path(k).name for k in skip_keys} if root is None else set()
        for nb in notebooks:
            nb_key = _notebook_key(nb, root)
            if nb_key in skip_keys:
                continue
            if root is None and nb.name in skip_names:
                continue
            remaining.append(nb)
        return remaining, prior

    if args.drive_folder_url:
        batch_start = args.drive_batch_start
        batch_size = args.drive_batch_size
        num_workers = args.workers or min(os.cpu_count() or 4, 8)
        if args.continuous and batch_size is None:
            raise ValueError("--continuous requires --drive-batch-size.")
        
        # Pre-flight scan to show user what will be processed
        print("\n--- Drive Folder Pre-Scan ---", file=sys.stderr)
        print("Scanning folder structure...", file=sys.stderr)
        try:
            scan_stats = scan_drive_folder(
                args.drive_folder_url,
                args.drive_credentials,
                include_subfolders=True,
            )
            print(
                f"✓ Found {scan_stats['total_notebooks']} notebooks across "
                f"{scan_stats['total_folders']} folders ({scan_stats['total_items']} total items)",
                file=sys.stderr
            )
            if batch_size:
                num_batches = (scan_stats['total_notebooks'] + batch_size - 1) // batch_size
                print(f"✓ Will process in {num_batches} batches of {batch_size} notebooks", file=sys.stderr)
            print("--- Starting Download ---\n", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Warning: Could not scan folder: {e}", file=sys.stderr)
        
        run_id = f"{run_timestamp}_drive"
        batch_num = 1
        # Accumulate all notebooks across batches into one dict
        all_per_notebook: Dict[Path, Tuple[List[str], List[str]]] = {}

        while True:
            print(f"\n[Batch {batch_num}] Downloading from Drive...", file=sys.stderr)
            drive_folder, kept = download_drive_folder(
                args.drive_folder_url,
                args.drive_credentials,
                batch_start=batch_start,
                batch_size=batch_size,
                include_subfolders=True,
            )
            print(f"[Batch {batch_num}] Downloaded {kept} notebooks", file=sys.stderr)
            notebooks = collect_notebooks(drive_folder)
            if not notebooks:
                raise FileNotFoundError(
                    f"No .ipynb files found in downloaded Drive folder: {drive_folder}"
                )
            notebooks, prior = _resolve_resume(notebooks, None)
            for nb, val in prior.items():
                if nb not in all_per_notebook:
                    all_per_notebook[nb] = val
            print(f"[Batch {batch_num}] Validating {len(notebooks)} notebooks ({num_workers} workers)...", file=sys.stderr)
            per_notebook = _parallel_validate_notebooks(
                notebooks, args.log_folder, run_tests,
                args.quiet, args.assert_checks,
                num_workers, args.timeout,
                checkpoint_path, args.checkpoint_every, None,
                per_notebook=all_per_notebook,
            )
            all_per_notebook.update(per_notebook)

            if not args.continuous:
                break
            batch_start += kept
            batch_num += 1
            if kept == 0 or (batch_size is not None and kept < batch_size):
                break

        # Write single .log and single .csv
        log_path = write_run_log(all_per_notebook, args.log_folder, "drive")
        csv_path = write_run_csv(all_per_notebook, args.log_folder)
        print(f"\nLog:  {log_path}")
        print(f"CSV:  {csv_path}")
        return 0

    if args.input_folder:
        num_workers = args.workers or min(os.cpu_count() or 4, 8)
        
        notebooks = collect_notebooks(args.input_folder)
        if not notebooks:
            raise FileNotFoundError(f"No .ipynb files found in {args.input_folder}")

        notebooks, prior = _resolve_resume(notebooks, args.input_folder)
        
        print(f"Validating {len(notebooks)} notebooks ({num_workers} workers)...", file=sys.stderr)
        per_notebook = _parallel_validate_notebooks(
            notebooks, args.log_folder, run_tests,
            args.quiet, args.assert_checks,
            num_workers, args.timeout,
            checkpoint_path, args.checkpoint_every, args.input_folder,
            per_notebook=prior,
        )

        # Write single .log and single .csv
        log_path = write_run_log(per_notebook, args.log_folder, "folder")
        csv_path = write_run_csv(per_notebook, args.log_folder)
        print(f"\nLog:  {log_path}")
        print(f"CSV:  {csv_path}")
        return 0

    if args.input_json_folder:
        json_files = sorted(args.input_json_folder.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {args.input_json_folder}"
            )
        print(f"Processing {len(json_files)} JSON files...")
        progress = ProgressTracker(len(json_files))
        for json_path in json_files:
            log_path, summary_path = validate_scicode_json(
                json_path,
                args.log_folder,
                run_tests,
                args.quiet,
                args.assert_checks,
            )
            progress.update()
        progress.finish()
        return 0

    if args.input_json:
        log_path, summary_path = validate_scicode_json(
            args.input_json,
            args.log_folder,
            run_tests,
            args.quiet,
            args.assert_checks,
        )
        return 0


    if not args.input_path:
        raise ValueError(
            "Provide input_path or --input-folder or --input-json or --drive-folder-url."
        )

    if args.input_path.suffix.lower() != ".ipynb":
        raise ValueError("Input file must have a .ipynb extension.")

    issues, results = validate_notebook(
        args.input_path,
        args.log_folder,
        run_tests,
        args.quiet,
        args.assert_checks,
    )

    # Single-file mode: still write one .log + one .csv
    per_notebook = {args.input_path: (issues, results)}
    log_path = write_run_log(per_notebook, args.log_folder, "single")
    csv_path = write_run_csv(per_notebook, args.log_folder)
    print(f"\nLog:  {log_path}")
    print(f"CSV:  {csv_path}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
