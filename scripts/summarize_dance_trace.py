#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(trace_dir: Path) -> list[tuple[Path, int, dict[str, Any]]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                records.append((path, lineno, json.loads(line)))
    return records


def summarize(records: list[tuple[Path, int, dict[str, Any]]]) -> str:
    event_counts: Counter[str] = Counter()
    file_event_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    nonfinite_records: list[str] = []
    actor_source_mismatches: list[str] = []
    trace_state: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "writer_files": set(),
            "start": 0,
            "end": 0,
            "error": 0,
            "reward": 0,
            "video_summary": 0,
            "rollout_steps": 0,
            "actor_start": 0,
            "actor_end": 0,
            "actor_error": 0,
        }
    )

    for path, lineno, record in records:
        event = str(record.get("event"))
        context = record.get("context", {})
        payload = record.get("payload", {})
        trace_id = context.get("trace_id")
        source_trace_id = context.get("source_trace_id")
        file_event_counts[path.name][event] += 1
        event_counts[event] += 1

        stats = payload.get("stats")
        if isinstance(stats, dict) and int(stats.get("nonfinite_count", 0)) > 0:
            nonfinite_records.append(f"{path.name}:{lineno} {event} trace={trace_id}")

        if event.startswith("actor_update"):
            writer_rank = context.get("rank")
            if source_trace_id:
                source_rank = context.get("source_rank")
                if writer_rank is not None and source_rank is not None and writer_rank != source_rank:
                    actor_source_mismatches.append(
                        f"{path.name}:{lineno} writer_rank={writer_rank} source_rank={source_rank} "
                        f"trace={trace_id} source_trace={source_trace_id}"
                    )
            elif trace_id and writer_rank is not None:
                trace_prefix = str(trace_id).split("_", 1)[0]
                if trace_prefix.startswith("rank") and trace_prefix != f"rank{writer_rank}":
                    actor_source_mismatches.append(
                        f"{path.name}:{lineno} writer_rank={writer_rank} legacy_trace={trace_id}"
                    )

        if trace_id:
            state = trace_state[trace_id]
            state["writer_files"].add(path.name)
            if event == "rollout_sample_start":
                state["start"] += 1
            elif event == "rollout_sample_end":
                state["end"] += 1
            elif event == "rollout_sample_error":
                state["error"] += 1
            elif event == "rollout_reward":
                state["reward"] += 1
            elif event == "rollout_video_summary":
                state["video_summary"] += 1
            elif event == "tensor_stats" and context.get("name") == "rollout.model_pred":
                state["rollout_steps"] += 1
            elif event == "actor_update_sample_start":
                state["actor_start"] += 1
            elif event == "actor_update_sample_end":
                state["actor_end"] += 1
            elif event == "actor_update_sample_error":
                state["actor_error"] += 1

    incomplete_rollouts = []
    incomplete_actor_updates = []
    for trace_id, state in sorted(trace_state.items()):
        if trace_id.startswith("rank") and "_rollout" in trace_id and (
            (state["start"] and not (state["end"] or state["error"]))
            or (state["rollout_steps"] and not (state["reward"] and state["video_summary"]))
        ):
            incomplete_rollouts.append(
                f"{trace_id} files={sorted(state['writer_files'])} "
                f"steps={state['rollout_steps']} reward={state['reward']} video={state['video_summary']}"
            )
        if trace_id.startswith("rank") and "_update" in trace_id and state["actor_start"] and not (
            state["actor_end"] or state["actor_error"]
        ):
            incomplete_actor_updates.append(f"{trace_id} files={sorted(state['writer_files'])}")

    lines = []
    lines.append("== Trace Files ==")
    for file_name in sorted(file_event_counts):
        lines.append(f"{file_name}: {dict(sorted(file_event_counts[file_name].items()))}")

    lines.append("")
    lines.append("== Event Counts ==")
    lines.append(str(dict(sorted(event_counts.items()))))

    lines.append("")
    lines.append("== Checks ==")
    lines.append(f"nonfinite_records={len(nonfinite_records)}")
    lines.append(f"actor_source_rank_mismatches={len(actor_source_mismatches)}")
    lines.append(f"incomplete_rollouts={len(incomplete_rollouts)}")
    lines.append(f"incomplete_actor_updates={len(incomplete_actor_updates)}")

    if nonfinite_records:
        lines.append("")
        lines.append("== Nonfinite Examples ==")
        lines.extend(nonfinite_records[:20])

    if actor_source_mismatches:
        lines.append("")
        lines.append("== Actor Source Mismatches ==")
        lines.extend(actor_source_mismatches[:20])

    if incomplete_rollouts:
        lines.append("")
        lines.append("== Incomplete Rollouts ==")
        lines.extend(incomplete_rollouts[:20])

    if incomplete_actor_updates:
        lines.append("")
        lines.append("== Incomplete Actor Updates ==")
        lines.extend(incomplete_actor_updates[:20])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dance trace jsonl files.")
    parser.add_argument("--trace-dir", default="dance_case4_traces", help="Directory containing *.jsonl trace files.")
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        raise SystemExit(f"trace dir not found: {trace_dir}")

    records = load_records(trace_dir)
    print(summarize(records))


if __name__ == "__main__":
    main()
