"""CLI wiring and orchestration for Lab10 -- Conversation Summarization &
Context Economics.

Commands (the `experiment` positional argument):
  - all: generate both summaries, run every follow-up question under all
    three conditions, evaluate quality, compute token economics, write
    reports and charts. This project's documented 14-call default
    (1 dataset x (2 summaries + 4x3 follow-ups)).
  - summarize: generate and inspect the selected summaries only (no
    downstream follow-up calls) -- plans and budgets only the
    summary-generation calls.
  - evaluate: re-run follow-up answering against an EXISTING, already
    generated summary artifact (loaded from a prior run's
    `generated_summaries.json`, found via --experiment-run-id) -- never
    regenerates a summary; fails clearly if the source run has none.
  - backfill: offline recomputation of reports/charts from persisted
    measurements. No API calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from lab_core.lab_cli_support import LiveExecutionNotAllowed, add_common_arguments, build_context
from labs.lab10_conversation_summarization.analysis import compute_reports, persist_computed_reports
from labs.lab10_conversation_summarization.charts_cli import add_backfill_arguments, run_backfill
from labs.lab10_conversation_summarization.datasets import DATASET_IDS, check_dataset_integrity, dataset_version_hashes, load_conversation, load_golden_questions
from labs.lab10_conversation_summarization.matrix import DEFAULT_QUESTION_COUNT, build_plan, datasets_for_profile, PROFILES
from labs.lab10_conversation_summarization.models import (
    CALL_PURPOSE_FOLLOWUP_ANSWER,
    CALL_PURPOSE_SUMMARY_GENERATION,
    CONDITION_FULL_HISTORY_BASELINE,
    CONDITIONS,
)
from labs.lab10_conversation_summarization.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab10_run_dir,
    read_generated_summaries,
    refuse_if_already_has_measurements,
    write_experiment_plan_csv,
    write_followup_measurements,
    write_generated_summaries,
    write_manifest,
    write_original_conversations,
    write_summary_measurements,
    write_traces,
)
from labs.lab10_conversation_summarization.prompts import ANSWER_PROMPT_HASH, SUMMARIZATION_PROMPT_HASH
from labs.lab10_conversation_summarization.runner import execute_plan

LAB_ID = "lab10"
_EXPERIMENTS = ["all", "summarize", "evaluate", "backfill"]

CONDITION_LABELS = {"full_history_baseline": "Full History", "summarization_low": "Summarization (Low)", "summarization_medium": "Summarization (Medium)"}


def add_lab10_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Must be reasoning-capable (see config lab10.model). Used for both summarization and answering.")
    parser.add_argument("--profile", default=None, choices=PROFILES, help="Dataset/question scope (default: config lab10.default_profile).")
    parser.add_argument("--dataset", default=None, choices=DATASET_IDS, help="Restrict the 'default' profile to one dataset (ignored by pilot/extended, which have fixed scopes).")
    parser.add_argument("--dry-run", action="store_true", help="Print and save the experiment plan, never call the API.")
    parser.add_argument("--show-io", action="store_true", help="Print the full request/response trace for every call, not just one representative example.")
    parser.add_argument("--experiment-run-id", default=None, help="Name this run. For 'evaluate', names the EXISTING run whose generated summaries will be reused.")


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab10-%Y%m%dT%H%M%SZ")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _print_plan_header(model, profile, datasets, question_count, total_planned, live_enabled) -> None:
    print("LAB10 -- Conversation Summarization & Context Economics\n")
    print(f"Model: {model}")
    print(f"Profile: {profile}\n")
    print(f"Datasets: {len(datasets)} ({', '.join(datasets)})")
    print(f"Follow-Up Questions per Dataset: {question_count}\n")
    print("Conditions:")
    for cid in CONDITIONS:
        print(f"  {cid}")
    print(f"\nPlanned API Calls: {total_planned}")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_before_after(dataset_id, question, generated_summaries, followups_by_condition) -> None:
    from labs.lab10_conversation_summarization.context import render_summary_as_context

    print("\n--- CONTEXT INSPECTION ---\n")
    print(f"Question: {question.question}\n")
    low_summary = generated_summaries.get((dataset_id, "summarization_low"))
    print("Generated Summary (Low):")
    print(render_summary_as_context(low_summary) if low_summary else "PENDING_GENERATION")
    print()
    for cid in CONDITIONS:
        m = next((fm for fm in followups_by_condition.get(cid, []) if fm.question_id == question.question_id), None)
        status = m.evaluation_status if m else "N/A"
        print(f"{CONDITION_LABELS[cid]} Answer Evaluation: {status}")
    print(f"Expected Answer: {question.expected_answer}")


def _print_grand_summary(computed) -> None:
    print("\n--- LAB10 GRAND TEST SUMMARY ---\n")
    for c in computed["conditions"]:
        print(f"{c['dataset_id']} / {CONDITION_LABELS.get(c['condition_id'], c['condition_id'])}:")
        if c["condition_id"] != CONDITION_FULL_HISTORY_BASELINE:
            print(f"  Summary Generation Cost:  {_fmt_cost(c['summary_generation_cost_usd'])}")
        print(f"  Follow-Up Cost:            {_fmt_cost(c['followup_cost_usd'])}")
        print(f"  Total Workflow Cost:       {_fmt_cost(c['total_workflow_cost_usd'])}")
        print(f"  Correct Answers:           {c['correct_answers']}/{c['evaluated_answers']}")
        if c["condition_id"] != CONDITION_FULL_HISTORY_BASELINE:
            print(f"  Context Reduction:         {_fmt_pct(c['mean_context_reduction_percent'] and c['mean_context_reduction_percent'] / 100)}")
            print(f"  Fact Retention Rate:       {_fmt_pct(c['fact_retention_rate'])}")
            print(f"  Known Fact Violations:     {c['known_fact_violation_count']}")
            print(f"  Net Savings:               {_fmt_cost(c['net_savings_usd'])}")
        print()

    from labs.lab10_conversation_summarization.analysis import break_even_summary_rows

    for b in break_even_summary_rows(computed["break_even"]):
        status = b["break_even_status"]
        label = f"follow-up #{b['break_even_call_index']}" if status == "REACHED" else "NOT REACHED"
        print(f"Break-Even [{b['dataset_id']} / {CONDITION_LABELS.get(b['condition_id'])}]: {label}")

    print("\nOverall Assessment:")
    print(
        "  Context compression, cumulative cost, break-even, answer correctness, fact retention, "
        "and reasoning overhead are reported above independently -- see analysis_report.md for the "
        "full written assessment. No single number picks a winner."
    )


def _conservative_dry_run_cost(plan, pricing, summarization_max_completion_tokens, answering_max_completion_tokens) -> "float | None":
    if pricing is None:
        return None
    conservative_prompt_tokens = 4000
    total = 0.0
    for call in plan:
        output_tokens = summarization_max_completion_tokens if call.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION else answering_max_completion_tokens
        total += (conservative_prompt_tokens / 1_000_000) * pricing.input_per_1m_usd + (output_tokens / 1_000_000) * pricing.output_per_1m_usd
    return total


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab10_config = probe_ctx.config.lab10
    if not lab10_config.enabled:
        print("Lab10 is disabled for this config (lab10.enabled = false).")
        return

    model = args.model or lab10_config.model
    profile = args.profile or lab10_config.default_profile
    default_dataset_ids = (args.dataset,) if (args.dataset and profile == "default") else (DATASET_IDS[0],)
    dataset_version = lab10_config.dataset_version

    datasets_in_scope = datasets_for_profile(profile, default_dataset_ids)
    for dataset_id in datasets_in_scope:
        problems = check_dataset_integrity(dataset_id, dataset_version)
        if problems:
            print(f"\nRefusing: dataset '{dataset_id}' (version {dataset_version}) failed integrity checks:")
            for p in problems:
                print(f"  - {p}")
            print("Nothing was executed.")
            return

    plan = build_plan(profile, default_dataset_ids, DEFAULT_QUESTION_COUNT, dataset_version)
    total_planned_all = len(plan)

    preloaded_summaries: dict = {}
    experiment_run_id = args.experiment_run_id or _generate_run_id()

    if args.experiment == "summarize":
        plan = [c for c in plan if c.call_purpose == CALL_PURPOSE_SUMMARY_GENERATION]
    elif args.experiment == "evaluate":
        if not args.experiment_run_id:
            print("\n'evaluate' requires --experiment-run-id naming an EXISTING run that already has generated summaries. Nothing was executed.")
            return
        source_run_dir = lab10_run_dir(probe_ctx.output_dir, args.experiment_run_id)
        preloaded_summaries = read_generated_summaries(source_run_dir)
        if not preloaded_summaries:
            print(f"\n'evaluate' found no generated summaries in run '{args.experiment_run_id}' -- cannot evaluate. "
                  "Run `lab10 summarize` or `lab10 all` first. Nothing was executed.")
            return
        plan = [c for c in plan if c.call_purpose == CALL_PURPOSE_FOLLOWUP_ANSWER]
        experiment_run_id = args.experiment_run_id + "-evaluate-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    total_planned = len(plan)
    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, profile, datasets_in_scope, 4 if profile != "pilot" else 2, total_planned, requested_live)

    run_dir = lab10_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, plan)
        pricing = probe_ctx.config.pricing.get(model)
        conservative_cost = _conservative_dry_run_cost(plan, pricing, lab10_config.summarization_max_completion_tokens, lab10_config.answering_max_completion_tokens)
        print(f"\nConservative Preflight Cost Ceiling (NOT a measured cost): {_fmt_cost(conservative_cost)}")
        print(f"Dry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab10 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab10_config.max_api_calls is not None and total_planned_all > lab10_config.max_api_calls:
        print(
            f"\nRefusing to execute: this profile's full plan ({total_planned_all} calls) exceeds "
            f"lab10.max_api_calls ({lab10_config.max_api_calls}). No API calls were made."
        )
        return

    try:
        refuse_if_already_has_measurements(run_dir)
    except RunIdAlreadyUsed as exc:
        print(f"\n{exc}")
        return

    try:
        live_ctx = build_context(requested_live=True, config_path=args.config)
    except LiveExecutionNotAllowed as exc:
        print(f"\n{exc}")
        return

    pricing = probe_ctx.config.pricing.get(model)
    if pricing is None:
        print(f"\nNo pricing configured for model '{model}': estimated_cost_usd will be None for every call.")
    print(f"[COST WARNING] {probe_ctx.config.execution.cost_warning_message}")

    conversations = {dataset_id: load_conversation(dataset_id, dataset_version) for dataset_id in datasets_in_scope}
    questions = {
        dataset_id: {q.question_id: q for q in load_golden_questions(dataset_id, dataset_version)}
        for dataset_id in datasets_in_scope
    }

    manifest = build_initial_manifest(
        experiment_run_id, lab10_config, profile, total_planned, dataset_version,
        dataset_version_hashes(dataset_version), {"summarization": SUMMARIZATION_PROMPT_HASH, "answer": ANSWER_PROMPT_HASH},
    )
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)
    write_original_conversations(run_dir, conversations)

    summary_measurements, followup_measurements, traces, generated_summaries = execute_plan(
        live_ctx.client, model, lab10_config.summarization_max_completion_tokens, model,
        lab10_config.answering_reasoning_effort, lab10_config.answering_max_completion_tokens,
        pricing, experiment_run_id, conversations, questions, plan, preloaded_summaries,
    )

    write_summary_measurements(run_dir, summary_measurements)
    write_followup_measurements(run_dir, followup_measurements)
    write_generated_summaries(run_dir, generated_summaries)
    write_traces(run_dir, traces)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)

    computed = compute_reports(manifest, summary_measurements, followup_measurements, generated_summaries, dataset_version)
    persist_computed_reports(run_dir, manifest, computed)

    if args.show_io:
        print("\n--- FULL REQUEST/RESPONSE TRACE ---")
        for t in traces:
            print(f"\n[{t.call_purpose}] dataset={t.dataset_id} condition={t.condition_id} question={t.question_id}")
            print(f"  request_parameters: {t.request_parameters}")
            print(f"  response_usage: {t.response_usage}")
            print(f"  response_visible_content: {t.response_visible_content}")
    elif followup_measurements:
        followups_by_condition: dict = {}
        for m in followup_measurements:
            followups_by_condition.setdefault(m.condition_id, []).append(m)
        first_dataset = datasets_in_scope[0]
        first_question = questions[first_dataset][followup_measurements[0].question_id]
        _print_before_after(first_dataset, first_question, generated_summaries, followups_by_condition)

    _print_grand_summary(computed)

    try:
        from labs.lab10_conversation_summarization.charts import generate_summarization_charts

        chart_paths = generate_summarization_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
    except Exception as exc:  # noqa: BLE001 -- a charting failure must never cost an already-persisted measurement.
        print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\nPlanned Calls:     {total_planned}")
    print(f"Results Directory: {run_dir}")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab10 - Conversation Summarization & Context Economics")
    parser.add_argument(
        "experiment", choices=_EXPERIMENTS,
        help="all runs the full experiment (summaries + every follow-up under every condition); "
        "summarize generates/inspects summaries only; evaluate re-runs follow-up answering against "
        "an existing summary artifact (--experiment-run-id); backfill recomputes reports/charts "
        "offline from existing results.",
    )
    add_common_arguments(parser)
    add_lab10_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
