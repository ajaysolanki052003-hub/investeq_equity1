#!/usr/bin/env python3
"""
Native messaging host for Chrome extension -> Python notebook validator.

Host name: com.scicode.validator.native
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import struct
import sys
import tempfile
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


if getattr(sys, "frozen", False):
    REPO_ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
else:
    REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validator import custom_notebook_checks as plugin_custom_checks  # noqa: E402
from src.validator import scicode_notebook_validator as notebook_validator  # noqa: E402

MAX_PREVIEW_LINES = 80
DEFAULT_LOG_CHUNK_SIZE = 180_000
MAX_LOG_CHUNK_SIZE = 500_000


def read_message() -> Optional[Dict]:
    raw_length = sys.stdin.buffer.read(4)
    if len(raw_length) == 0:
        return None
    if len(raw_length) != 4:
        return None
    message_length = struct.unpack("<I", raw_length)[0]
    message_bytes = sys.stdin.buffer.read(message_length)
    if len(message_bytes) != message_length:
        return None
    return json.loads(message_bytes.decode("utf-8"))


def send_message(payload: Dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(encoded)))
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.flush()


def get_report_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path.home() / "SciCodeNative" / "output" / "native_backend_logs"
    return REPO_ROOT / "output" / "native_backend_logs"


def summarize_result_counts(result_lines: List[str]) -> Tuple[int, int, int]:
    pass_count = 0
    fail_count = 0
    for line in result_lines:
        stripped = str(line).strip()
        if stripped.startswith("PASS "):
            pass_count += 1
        elif stripped.startswith("FAIL "):
            fail_count += 1
    return pass_count, fail_count, len(result_lines)


def read_log_chunk_message(message: Dict) -> Dict:
    raw_path = str(message.get("log_path", "")).strip()
    if not raw_path:
        return {"ok": False, "error": "Missing log_path for read_log_chunk action."}

    try:
        log_path = Path(raw_path).resolve()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Invalid log_path: {type(exc).__name__}: {exc}"}

    report_dir = get_report_dir().resolve()
    try:
        log_path.relative_to(report_dir)
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "read_log_chunk rejected path outside allowed report directory."}

    if not log_path.exists():
        return {"ok": False, "error": f"Log file not found: {log_path}"}

    try:
        offset = int(message.get("offset", 0))
    except Exception:  # noqa: BLE001
        offset = 0
    offset = max(0, offset)

    try:
        chunk_size = int(message.get("chunk_size", DEFAULT_LOG_CHUNK_SIZE))
    except Exception:  # noqa: BLE001
        chunk_size = DEFAULT_LOG_CHUNK_SIZE
    chunk_size = max(1_000, min(chunk_size, MAX_LOG_CHUNK_SIZE))

    try:
        data = log_path.read_bytes()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"Failed to read log file: {type(exc).__name__}: {exc}"}

    if offset >= len(data):
        return {"ok": True, "content": "", "next_offset": len(data), "eof": True}

    end = min(offset + chunk_size, len(data))
    chunk = data[offset:end].decode("utf-8", errors="replace")
    return {
        "ok": True,
        "content": chunk,
        "next_offset": end,
        "eof": end >= len(data),
    }


def normalize_heading(text: str) -> str:
    return text.strip().lower()


def normalize_token(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def join_source(source) -> str:
    if isinstance(source, list):
        return "".join(source)
    return str(source or "")


def is_metadata_heading_line(line: str) -> bool:
    normalized = normalize_token(line)
    if normalized in {
        "domain:",
        "subdomain:",
        "references:",
        "perplexity link:",
        "**rlhf link**",
        "rlhf link",
        "main problem:",
        "**model failure report**",
        "model failure report",
        "*gpt-5.4*",
        "gpt-5.4",
        "*gemini-3.1-pro-preview*",
        "gemini-3.1-pro-preview",
        "*claude-opus-4.6*",
        "claude-opus-4.6",
        "main problem: pass/fail",
    }:
        return True
    if normalized.startswith("subproblem "):
        # Supports both "Subproblem N:" and "Subproblem N: PASS/FAIL" templates.
        return True
    return False


def extract_metadata_value(lines: List[str], key_label: str) -> Optional[str]:
    key = normalize_token(key_label)
    for i, line in enumerate(lines):
        line_norm = normalize_token(line)
        if not (line_norm == key or (key.endswith(":") and line_norm.startswith(key))):
            continue
        # Inline value format: "Field: value"
        colon_index = str(line).find(":")
        if colon_index >= 0:
            inline = str(line)[colon_index + 1 :].strip()
            # Treat "PASS/FAIL" marker as placeholder (not a concrete value).
            if inline and normalize_token(inline) != "pass/fail":
                return inline
        for j in range(i + 1, len(lines)):
            candidate = lines[j].strip()
            if not candidate:
                continue
            if is_metadata_heading_line(candidate):
                return ""
            return candidate
        return ""
    return None


def validate_metadata_template(metadata_text: str, subproblem_count: int) -> List[str]:
    issues: List[str] = []
    raw_lines = [line.strip() for line in metadata_text.splitlines() if line.strip()]
    model_headings = ["*GPT-5.4*", "*Gemini-3.1-pro-preview*", "*Claude-opus-4.6*"]
    model_heading_norms = {heading.lower() for heading in model_headings}

    required_value_fields = [
        "Domain:",
        "Subdomain:",
        "References:",
        "Perplexity Link:",
        "Main Problem:",
    ]
    required_value_fields.extend(
        [f"Subproblem {idx}:" for idx in range(1, max(subproblem_count, 0) + 1)]
    )
    required_presence_only = [
        ["**RLHF LINK**", "RLHF LINK"],
        ["**Model Failure Report**", "Model Failure Report"],
        [model_headings[0], "GPT-5.4"],
        [model_headings[1], "Gemini-3.1-pro-preview"],
        [model_headings[2], "Claude-opus-4.6"],
    ]
    # Accept natural status labels (e.g. "Subproblem 1: PASS") instead of requiring literal "PASS/FAIL" in key.
    model_status_fields = [f"Subproblem {idx}:" for idx in range(1, max(subproblem_count, 0) + 1)]
    model_status_fields.append("Main Problem:")
    empty_placeholders = {"pass/fail", "tbd", "todo", "-", "na", "n/a"}

    for field in required_value_fields:
        value = extract_metadata_value(raw_lines, field)
        if value is None:
            issues.append(f"Metadata missing required field '{field}'.")
        elif not value:
            issues.append(f"Metadata field '{field}' is present but value is empty.")
        elif value.strip().lower() in empty_placeholders:
            issues.append(f"Metadata field '{field}' has placeholder value '{value}'.")

    for variants in required_presence_only:
        variant_norms = {normalize_token(v) for v in variants}
        present = any(normalize_token(line) in variant_norms for line in raw_lines)
        if not present:
            issues.append(f"Metadata missing required field '{variants[0]}'.")

    for model_heading in model_headings:
        try:
            model_idx = next(
                i for i, line in enumerate(raw_lines) if line.strip().lower() == model_heading.lower()
            )
        except StopIteration:
            continue
        next_model_idx = None
        for i in range(model_idx + 1, len(raw_lines)):
            if raw_lines[i].strip().lower() in model_heading_norms:
                next_model_idx = i
                break
        model_block = raw_lines[model_idx + 1 : (next_model_idx if next_model_idx is not None else len(raw_lines))]
        for status_field in model_status_fields:
            value = extract_metadata_value(model_block, status_field)
            if value is None:
                issues.append(
                    f"Metadata '{model_heading}' missing required status line '{status_field}'."
                )
            elif not value:
                issues.append(
                    f"Metadata '{model_heading}' status '{status_field}' has empty value."
                )
            elif value.strip().lower() in empty_placeholders:
                issues.append(
                    f"Metadata '{model_heading}' status '{status_field}' has placeholder value '{value}'."
                )

    return issues


def analyze_notebook_cells_and_metadata(notebook_json: Dict) -> List[str]:
    issues: List[str] = []
    cells = notebook_json.get("cells", [])
    section_cell_counts: Dict[str, int] = {}
    current_top_section: Optional[str] = None
    metadata_text = ""
    subproblem_headings: set[str] = set()
    unscoped_cells = 0

    def is_recognized_top_section(heading: str) -> bool:
        return heading in {"title", "metadata", "main problem"} or heading.startswith("subproblem")

    for cell in cells:
        cell_type = cell.get("cell_type")
        text = join_source(cell.get("source", [])).strip()
        if cell_type == "markdown" and text.startswith("# "):
            heading = normalize_heading(text[2:])
            if is_recognized_top_section(heading):
                current_top_section = heading
                if heading.startswith("subproblem"):
                    subproblem_headings.add(heading)
            else:
                current_top_section = None

        if current_top_section:
            section_cell_counts[current_top_section] = section_cell_counts.get(current_top_section, 0) + 1
        else:
            unscoped_cells += 1

        if current_top_section == "metadata" and cell_type == "markdown":
            metadata_text = f"{metadata_text}\n{text}" if metadata_text else text

    subproblem_count = len(subproblem_headings)

    expected_cells_per_problem_section = 9
    expected_title_cells = 2
    expected_metadata_cells = 2
    expected_total = expected_title_cells + expected_metadata_cells + (subproblem_count + 1) * expected_cells_per_problem_section
    actual_total = len(cells)

    problem_sections = ["main problem"] + sorted(
        [k for k in section_cell_counts if k.startswith("subproblem")],
        key=lambda name: int("".join(ch for ch in name if ch.isdigit()) or "9999"),
    )
    for section_name in problem_sections:
        count = section_cell_counts.get(section_name, 0)
        if count != expected_cells_per_problem_section:
            issues.append(
                f"Section '{section_name}' has {count} cells; expected exactly {expected_cells_per_problem_section}."
            )

    if "title" in section_cell_counts:
        if section_cell_counts["title"] != expected_title_cells:
            issues.append(
                f"Section 'title' has {section_cell_counts['title']} cells; expected exactly {expected_title_cells}."
            )
    if "metadata" in section_cell_counts:
        if section_cell_counts["metadata"] != expected_metadata_cells:
            issues.append(
                f"Section 'metadata' has {section_cell_counts['metadata']} cells; expected exactly {expected_metadata_cells}."
            )

    if actual_total > expected_total:
        issues.append(
            f"Notebook has {actual_total} total cells; expected {expected_total} for {subproblem_count} subproblem(s). Extra cells detected."
        )
    elif actual_total < expected_total:
        issues.append(
            f"Notebook has {actual_total} total cells; expected {expected_total} for {subproblem_count} subproblem(s). Missing cells detected."
        )

    if unscoped_cells > 0:
        issues.append(f"{unscoped_cells} cell(s) are outside recognized top-level sections.")

    if metadata_text.strip():
        issues.extend(validate_metadata_template(metadata_text, subproblem_count))
    else:
        issues.append("Metadata content block is empty or not found.")

    return issues


def extract_declared_functions_from_prompt(prompt_text: str) -> List[str]:
    """Extract top-level function names from prompt fenced code blocks."""
    names: List[str] = []
    if not prompt_text:
        return names
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", prompt_text, re.DOTALL)
    for block in code_blocks:
        names.extend(notebook_validator.extract_functions(block, top_level_only=True))
    # Preserve first-seen order while deduping.
    return list(dict.fromkeys(names))


def validate_subproblem_headers_present_in_main_solution(
    sections: Dict[str, Dict[str, str]]
) -> List[str]:
    """
    Ensure every subproblem prompt-declared function exists in main solution code.
    """
    issues: List[str] = []
    main_solution = sections.get("main problem", {}).get("solution", "")
    if not main_solution:
        return issues

    main_solution_functions = set(
        notebook_validator.extract_functions(main_solution, top_level_only=True)
    )
    for section_name, data in sections.items():
        if not section_name.startswith("subproblem"):
            continue
        prompt_text = data.get("prompt", "")
        declared_funcs = extract_declared_functions_from_prompt(prompt_text)
        for fn_name in declared_funcs:
            if fn_name not in main_solution_functions:
                issues.append(
                    f"Main problem solution is missing subproblem function '{fn_name}' declared in '{section_name}' prompt."
                )
    return issues


def validate_notebook_message(message: Dict) -> Dict:
    notebook_json = message.get("notebook_json")
    source_url = str(message.get("source_url", "")).strip()
    run_tests = bool(message.get("run_tests", True))
    assert_checks = bool(message.get("assert_checks", True))
    if not isinstance(notebook_json, dict):
        return {"ok": False, "error": "Invalid notebook_json payload."}

    with tempfile.TemporaryDirectory(prefix="scicode_native_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input_notebook.ipynb"
        log_dir = tmp_dir / "logs"
        input_path.write_text(json.dumps(notebook_json, ensure_ascii=False, indent=2), encoding="utf-8")

        custom_issues = plugin_custom_checks.run_custom_checks_on_notebook_json(notebook_json)

        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                issues, results = notebook_validator.validate_notebook(
                    input_path=input_path,
                    log_folder=log_dir,
                    run_tests=run_tests,
                    quiet=True,
                    assert_checks=assert_checks,
                )
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"Python validator failed: {type(exc).__name__}: {exc}"}

        result_lines = list(results)
        all_issues = [f"[CUSTOM] {i}" for i in custom_issues] + list(issues)
        fail_lines = [
            line for line in (all_issues + result_lines)
            if isinstance(line, str) and line.strip().startswith("FAIL ")
        ]
        overall_status = "FAIL" if all_issues or fail_lines else "PASS"

        report_dir = get_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_log = report_dir / f"{ts}_native_validator_report.log"
        report_lines = [
            "=" * 72,
            "SCICODE PYTHON VALIDATOR REPORT (NATIVE BACKEND)",
            "=" * 72,
            f"Source URL: {source_url or '(not provided)'}",
            f"Run tests: {'yes' if run_tests else 'no'}",
            f"Assert checks: {'yes' if assert_checks else 'no'}",
            f"Overall: {overall_status}",
            "",
            f"Issues ({len(all_issues)}):",
        ]
        report_lines.extend([f"- {item}" for item in all_issues] if all_issues else ["- None"])
        report_lines.append("")
        report_lines.append(f"Results ({len(result_lines)}):")
        report_lines.extend([f"- {item}" for item in result_lines] if result_lines else ["- None"])
        report_lines.append("")
        captured_text = captured.getvalue().strip()
        if captured_text:
            report_lines.append("Captured validator output:")
            report_lines.append(captured_text)
            report_lines.append("")
        output_log.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

        pass_count, fail_count, result_count = summarize_result_counts(result_lines)
        issues_preview = [str(item) for item in all_issues[:MAX_PREVIEW_LINES]]
        results_preview = [str(item) for item in result_lines[:MAX_PREVIEW_LINES]]
        preview_truncated = len(all_issues) > len(issues_preview) or len(result_lines) > len(results_preview)

        return {
            "ok": True,
            "overall_status": overall_status,
            "issue_count": len(all_issues),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "result_count": result_count,
            "issues_preview": issues_preview,
            "results_preview": results_preview,
            "preview_truncated": preview_truncated,
            "log_path": str(output_log),
        }


def handle_message(message: Dict) -> Dict:
    action = message.get("action")
    if action == "ping":
        return {"ok": True, "message": "pong"}
    if action == "read_log_chunk":
        return read_log_chunk_message(message)
    if action == "validate_notebook":
        return validate_notebook_message(message)
    return {"ok": False, "error": f"Unknown action: {action}"}


def main() -> int:
    while True:
        message = read_message()
        if message is None:
            break
        response = handle_message(message)
        send_message(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
