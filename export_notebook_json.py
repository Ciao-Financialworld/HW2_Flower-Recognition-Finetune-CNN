#!/usr/bin/env python3
"""
导出 SwanLab 本地实验结果为 notebook 友好的 JSON 文件。

默认行为：
1. 扫描 ./swanlab_logs 下的所有 run 目录
2. 为每个实验导出一个 JSON 文件
3. 额外生成一个 manifest.json，汇总全部实验的关键信息

示例：
    python export_notebook_json.py
    python export_notebook_json.py --project flower102-task1
    python export_notebook_json.py --include-system
    python export_notebook_json.py --no-history
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./swanlab_logs"),
        help="SwanLab 本地日志根目录",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./exports/notebook_json"),
        help="JSON 导出目录",
    )
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="只导出指定 project，可重复传入多次",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="同时导出 GPU / CPU 等系统指标",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="不导出完整曲线，仅保留 summary",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_scalar_value(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    try:
        if "." in text or "e" in lowered:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_config_values(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        flat = {}
        for key, value in data.items():
            if isinstance(value, dict) and "value" in value:
                flat[key] = value["value"]
            else:
                flat[key] = value
        return flat
    except Exception:
        pass

    flat: dict[str, Any] = {}
    current_key: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_key = line[:-1].strip()
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.startswith("value:"):
            flat[current_key] = parse_scalar_value(stripped.split(":", 1)[1])
    return flat


def extract_json_records(raw: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    marker = '{"model_type":'
    text = raw.decode("utf-8", errors="ignore")

    for line in text.splitlines():
        start = line.find(marker)
        if start < 0:
            continue
        candidate = line[start:].strip()
        try:
            records.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue

    return records


def summarize_metric(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}

    values = [point["value"] for point in history if isinstance(point.get("value"), (int, float))]
    summary: dict[str, Any] = {"count": len(history)}

    last = history[-1]
    summary["final"] = last.get("value")
    summary["final_step"] = last.get("step")
    summary["final_epoch"] = last.get("epoch")

    if values:
        max_point = max(history, key=lambda x: x["value"])
        min_point = min(history, key=lambda x: x["value"])
        summary["max"] = max_point["value"]
        summary["max_step"] = max_point.get("step")
        summary["max_epoch"] = max_point.get("epoch")
        summary["min"] = min_point["value"]
        summary["min_step"] = min_point.get("step")
        summary["min_epoch"] = min_point.get("epoch")

    return summary


def build_run_payload(
    run_dir: Path,
    include_system: bool,
    include_history: bool,
) -> dict[str, Any]:
    files_dir = run_dir / "files"
    backup_path = run_dir / "backup.swanlab"

    records = extract_json_records(backup_path.read_bytes())
    config = load_config_values(files_dir / "config.yaml")
    metadata = load_json(files_dir / "swanlab-metadata.json")

    project: dict[str, Any] = {}
    experiment: dict[str, Any] = {}
    columns: dict[str, dict[str, Any]] = {}
    metric_history: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        model_type = record.get("model_type")
        data = record.get("data", {})

        if model_type == "Project":
            project = data
        elif model_type == "Experiment":
            experiment = data
        elif model_type == "Column":
            columns[data["key"]] = data
        elif model_type == "Scalar":
            key = data["key"]
            column = columns.get(key, {})
            is_system = column.get("cls") == "SYSTEM" or key.startswith("__swanlab__.")
            if is_system and not include_system:
                continue

            points = metric_history.setdefault(key, [])
            metric = data.get("metric", {})
            points.append(
                {
                    "step": data.get("step"),
                    "epoch": data.get("epoch"),
                    "value": metric.get("data"),
                    "create_time": metric.get("create_time"),
                }
            )

    metrics: dict[str, Any] = {}
    for key, history in sorted(metric_history.items()):
        column = columns.get(key, {})
        item: dict[str, Any] = {
            "summary": summarize_metric(history),
            "column": {
                "key": key,
                "class": column.get("cls"),
                "section_name": column.get("section_name"),
                "section_type": column.get("section_type"),
                "chart_name": column.get("chart_name"),
                "chart_reference": column.get("chart_reference"),
            },
        }
        if include_history:
            item["history"] = history
        metrics[key] = item

    if not project.get("name"):
        project["name"] = config.get("project")

    if not experiment.get("name"):
        experiment["name"] = config.get("model")

    if not experiment.get("name") and config.get("analysis") == "batch":
        batch_size = config.get("batch_size")
        if batch_size is not None:
            experiment["name"] = f"batch{batch_size}"

    if not experiment.get("id"):
        suffix = run_dir.name.split("-", 2)
        experiment["id"] = suffix[-1] if suffix else run_dir.name

    val_acc_summary = metrics.get("val/accuracy", {}).get("summary", {})
    test_acc_summary = metrics.get("test/accuracy", {}).get("summary", {})

    payload = {
        "project": project,
        "experiment": experiment,
        "run_dir": run_dir.name,
        "run_path": str(run_dir.resolve()),
        "config": config,
        "metadata": metadata,
        "metrics": metrics,
        "summary": {
            "project": project.get("name"),
            "experiment_name": experiment.get("name"),
            "run_id": experiment.get("id"),
            "best_val_accuracy": val_acc_summary.get("max"),
            "best_val_epoch": val_acc_summary.get("max_epoch"),
            "test_accuracy": test_acc_summary.get("final"),
            "metric_count": len(metrics),
        },
    }
    return payload


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "unknown"


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        path for path in log_dir.iterdir()
        if path.is_dir() and path.name.startswith("run-") and (path / "backup.swanlab").exists()
    )

    if not run_dirs:
        raise SystemExit(f"未找到实验目录：{log_dir}")

    wanted_projects = set(args.project)
    manifest: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        payload = build_run_payload(
            run_dir=run_dir,
            include_system=args.include_system,
            include_history=not args.no_history,
        )

        project_name = payload["project"].get("name", "")
        if wanted_projects and project_name not in wanted_projects:
            continue

        exp_name = payload["experiment"].get("name", "unknown")
        run_id = payload["experiment"].get("id", run_dir.name)
        file_name = f"{safe_name(project_name)}__{safe_name(exp_name)}__{safe_name(run_id)}.json"
        out_path = out_dir / file_name

        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest.append(
            {
                **payload["summary"],
                "file": file_name,
            }
        )
        print(f"[OK] {project_name} / {exp_name} -> {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n导出完成：{len(manifest)} 个实验")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
