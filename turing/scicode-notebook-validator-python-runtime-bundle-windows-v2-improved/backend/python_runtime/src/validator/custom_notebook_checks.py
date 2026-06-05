#!/usr/bin/env python3
"""
Custom notebook checks originally used by native plugin backend.

These checks are intentionally separate from the core notebook validator so they
can be reused from CLI scripts, desktop app pipeline, and native host.
"""

from __future__ import annotations

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.validator import scicode_notebook_validator as notebook_validator


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
        return True
    return False


def extract_metadata_value(lines: List[str], key_label: str) -> Optional[str]:
    key = normalize_token(key_label)
    for i, line in enumerate(lines):
        line_norm = normalize_token(line)
        if not (line_norm == key or (key.endswith(":") and line_norm.startswith(key))):
            continue
        colon_index = str(line).find(":")
        if colon_index >= 0:
            inline = str(line)[colon_index + 1 :].strip()
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

    if "title" in section_cell_counts and section_cell_counts["title"] != expected_title_cells:
        issues.append(
            f"Section 'title' has {section_cell_counts['title']} cells; expected exactly {expected_title_cells}."
        )
    if "metadata" in section_cell_counts and section_cell_counts["metadata"] != expected_metadata_cells:
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
    names: List[str] = []
    if not prompt_text:
        return names
    code_blocks = re.findall(r"```[\w]*\n?(.*?)```", prompt_text, re.DOTALL)
    for block in code_blocks:
        names.extend(notebook_validator.extract_functions(block, top_level_only=True))
    return list(dict.fromkeys(names))


def validate_subproblem_headers_present_in_main_solution(
    sections: Dict[str, Dict[str, str]]
) -> List[str]:
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


def run_custom_checks_on_notebook_json(notebook_json: Dict) -> List[str]:
    issues: List[str] = []
    issues.extend(analyze_notebook_cells_and_metadata(notebook_json))
    try:
        with tempfile.TemporaryDirectory(prefix="scicode_custom_checks_") as tmp:
            tmp_path = Path(tmp) / "input_notebook.ipynb"
            tmp_path.write_text(json.dumps(notebook_json, ensure_ascii=False, indent=2), encoding="utf-8")
            sections, _parse_issues, _metadata_found, _title_found = notebook_validator.parse_notebook(tmp_path)
        issues.extend(validate_subproblem_headers_present_in_main_solution(sections))
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Failed to run subproblem-header consistency check: {type(exc).__name__}: {exc}")
    return issues


def run_custom_checks_from_file(notebook_path: Path) -> List[str]:
    with notebook_path.open("r", encoding="utf-8") as f:
        notebook_json = json.load(f)
    if not isinstance(notebook_json, dict):
        return ["Notebook is not a valid JSON object."]
    return run_custom_checks_on_notebook_json(notebook_json)


def write_custom_checks_report(
    notebook_path: Path,
    issues: List[str],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{ts}_custom_notebook_checks.log"
    lines = [
        "=" * 72,
        "SCICODE CUSTOM NOTEBOOK CHECKS REPORT",
        "=" * 72,
        f"Notebook: {notebook_path}",
        f"Issues: {len(issues)}",
        "",
        "Custom issues:",
    ]
    if issues:
        lines.extend([f"- {item}" for item in issues])
    else:
        lines.append("- None")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
