"""Read-only visual explorer for local experiment results.

Launch with:
    uv run --with "streamlit>=1.40,<2" streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from lab_core.dashboard_data import (
    EXPORT_COLUMNS,
    export_csv,
    filter_records,
    scan_results,
    summarize,
    summarize_lab07,
)
from lab_core.dashboard_view import (
    LAB_CATALOG,
    lab_measurement_counts,
    scenario_label,
    select_simulation_comparison,
    simulation_chart_rows,
    strategy_label,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(
    page_title="LLM Token Economics | Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("LLM Token Economics")
st.caption("Local experiment analytics · read-only · no provider calls")

records, warnings = scan_results(RESULTS_DIR)

with st.sidebar:
    st.header("Explore experiments")
    st.caption("Read-only view of saved results")
    if st.button("Refresh results", use_container_width=True):
        st.rerun()

    counts = lab_measurement_counts(records)
    labs = st.multiselect(
        "Lab",
        LAB_CATALOG,
        format_func=lambda lab: (
            f"{lab.upper()} · {counts[lab]} saved measurements"
            if counts[lab] else f"{lab.upper()} · No saved results"
        ),
        help="All ten labs are listed. Only experiments with local saved results have measurements.",
    )
    st.caption(
        f"Saved measurements found for "
        f"{sum(count > 0 for count in counts.values())} of {len(LAB_CATALOG)} labs."
    )
    if records:
        lab_rows = filter_records(records, labs=labs)
        runs = st.multiselect("Run", sorted({row["run_id"] for row in lab_rows}))
        run_rows = filter_records(lab_rows, runs=runs)
        models = st.multiselect("Model", sorted({row["model"] for row in run_rows}))
        model_rows = filter_records(run_rows, models=models)
        conditions = st.multiselect(
            "Strategy / condition",
            sorted({row["condition"] for row in model_rows}),
        )
        condition_rows = filter_records(model_rows, conditions=conditions)
        scenarios = st.multiselect(
            "Simulation scenario",
            sorted({
                row["scenario"] for row in condition_rows
                if row["lab"] == "lab07"
            }),
            format_func=scenario_label,
        )
    else:
        runs, models, conditions, scenarios = [], [], [], []

    with st.expander("Local data directory", expanded=False):
        st.code(str(RESULTS_DIR))

if warnings:
    with st.expander(f"Unreadable files ({len(warnings)})", expanded=False):
        for warning in warnings:
            st.warning(warning)

if not records:
    st.warning("No saved experiment results were found in the local results/ directory.")
    st.markdown("To create an API-free Lab07 simulation, run:")
    st.code("uv run lab-cli lab07 all", language="powershell")
    st.caption("Then select Refresh results. No API key is required.")
    st.stop()

filtered = filter_records(
    records, labs=labs, runs=runs, models=models,
    conditions=conditions, scenarios=scenarios,
)
if not filtered:
    if labs and not any(row["lab"] in labs for row in records):
        st.info(
            "No saved results exist yet for the selected lab(s) in this local "
            "results/ folder. The laboratory code is available, but selecting "
            "a lab does not run experiments. Most labs require a separate, "
            "explicitly authorized live API run to create measurements."
        )
    else:
        st.warning("No measurements match these filters. Clear or change the selections.")
    st.stop()

st.info(
    "Results are read from local files. Costs are estimates, not invoices; "
    "missing data is not interpreted as zero. Simulation results are distinct "
    "from real model observations.",
    icon="ℹ️",
)

costs = [
    row["estimated_cost_usd"] for row in filtered
    if row["estimated_cost_usd"] is not None
]
latencies = [
    row["latency_ms"] for row in filtered
    if row["latency_ms"] is not None
]
correctness = [
    row["answer_correct"] for row in filtered
    if row["answer_correct"] is not None
]
known_success = sum(row["succeeded"] is not None for row in filtered)
known_tokens = sum(row["total_tokens"] is not None for row in filtered)

a, b, c, d = st.columns(4)
a.metric("Measurements", f"{len(filtered):,}")
b.metric("Estimated cost", f"USD {sum(costs):.6f}" if costs else "N/A")
c.metric(
    "Mean measured latency",
    f"{sum(latencies) / len(latencies):,.1f} ms" if latencies else "N/A",
)
d.metric(
    "Evaluated answer accuracy",
    f"{sum(correctness) / len(correctness) * 100:.1f}%"
    if correctness else "N/A",
)
st.caption(
    f"Metric coverage · cost {len(costs)}/{len(filtered)} · "
    f"latency {len(latencies)}/{len(filtered)} · "
    f"answer quality {len(correctness)}/{len(filtered)} · "
    f"request success {known_success}/{len(filtered)} · "
    f"token usage {known_tokens}/{len(filtered)}"
)

if all(row["simulated"] for row in filtered):
    st.caption(
        "Simulation-only view: real LLM cost, token usage, model accuracy "
        "and actual request latency are unavailable by design."
    )

overview, comparison, simulator, detail = st.tabs(
    ["Overview", "Model comparison", "Retry simulator", "Measurements"]
)

with overview:
    st.subheader("Experiment runs")
    st.dataframe(
        summarize(filtered, group_keys=("lab", "experiment", "run_id")),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Sums include known measurements only. Coverage counts in the table "
        "show which observations contributed to each metric."
    )

with comparison:
    st.subheader("Model and condition comparison")
    actual_rows = [row for row in filtered if not row["simulated"]]
    if not actual_rows:
        st.info(
            "The selected records contain only simulator data. "
            "Open the Retry simulator tab for meaningful strategy comparisons."
        )
    else:
        st.caption(
            "Compare matched workloads within one lab and run. "
            "These descriptive differences do not establish causality."
        )
        run_choices = sorted({
            (row["lab"], row["run_id"]) for row in actual_rows
        })
        chosen_lab, chosen_run = st.selectbox(
            "Choose a lab and experiment run",
            run_choices,
            format_func=lambda value: f"{value[0]} · {value[1]}",
        )
        matched = [
            row for row in actual_rows
            if row["lab"] == chosen_lab and row["run_id"] == chosen_run
        ]
        groups = summarize(matched, group_keys=("model", "condition"))
        st.dataframe(groups, use_container_width=True, hide_index=True)
        chart_specs = (
            ("Estimated cost (known values)", "estimated_cost_usd", "USD"),
            ("Mean observed latency", "mean_latency_ms", "Milliseconds"),
            ("Measured token totals", "total_tokens", "Tokens"),
        )
        plotted = False
        for title, metric, unit in chart_specs:
            chart = [
                {
                    "Model / condition": f"{row['model']} / {row['condition']}",
                    unit: row[metric],
                }
                for row in groups if row[metric] is not None
            ]
            if chart:
                plotted = True
                st.markdown(f"**{title}**")
                st.bar_chart(
                    chart, x="Model / condition", y=unit,
                    horizontal=True, height=max(220, 45 * len(chart)),
                )
        if not plotted:
            st.info("This experiment contains no measured numeric values for these charts.")

with simulator:
    st.subheader("Lab07 · Retry strategy comparison")
    st.caption(
        "Each graph compares strategies within ONE simulated scenario and "
        "ONE run. The bars show virtual-clock outcomes, not real API calls, "
        "model costs, or response latency."
    )
    simulation = summarize_lab07(filtered)
    if not simulation:
        st.info("No Lab07 simulator measurements match your filters.")
    else:
        run_options = sorted({row["run_id"] for row in simulation})
        chosen_run = st.selectbox(
            "Simulation run", run_options, key="simulation_run",
        )
        scenario_options = sorted({
            row["scenario"] for row in simulation
            if row["run_id"] == chosen_run
        })
        chosen_scenario = st.selectbox(
            "Scenario", scenario_options,
            format_func=scenario_label, key="simulation_scenario",
        )
        selected = select_simulation_comparison(
            simulation, run_id=chosen_run, scenario=chosen_scenario,
        )
        known_clients = sum(
            row["simulated_clients"] for row in selected
        )
        known_success_count = sum(
            row["successful_clients"] for row in selected
            if row["successful_clients"] is not None
        )
        count_evaluated = sum(
            row["success_evaluated"] for row in selected
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Strategies", str(len(selected)))
        m2.metric("Simulated client outcomes", str(known_clients))
        m3.metric(
            "Successful outcomes",
            f"{known_success_count} / {count_evaluated}"
            if count_evaluated else "N/A",
        )
        table_rows = [
            {
                "Strategy": strategy_label(row["strategy"]),
                "Success (%)": row["success_rate_percent"],
                "Clients": row["simulated_clients"],
                "Evaluated": row["success_evaluated"],
                "Retries": row["total_retries"],
                "HTTP 429": row["total_429_responses"],
                "Virtual wait (s)": row["total_virtual_wait_seconds"],
            }
            for row in selected
        ]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        chart_pairs = (
            (
                ("Simulated success rate", "success_rate_percent", "Success (%)"),
                ("Total retries", "total_retries", "Retries"),
            ),
            (
                ("Simulated HTTP 429 responses", "total_429_responses", "429 responses"),
                ("Scheduled virtual wait", "total_virtual_wait_seconds", "Virtual wait (s)"),
            ),
        )
        for pair in chart_pairs:
            left, right = st.columns(2, gap="large")
            for column, (title, metric, unit) in zip((left, right), pair):
                with column:
                    st.markdown(f"**{title}**")
                    chart_rows = simulation_chart_rows(selected, metric, unit)
                    if chart_rows:
                        st.bar_chart(
                            chart_rows, x="Strategy", y=unit,
                            horizontal=True, height=260,
                        )
                    else:
                        st.caption("No recorded measurements for this metric.")
        st.caption(
            "Reading the charts: each strategy is evaluated under the "
            "same selected scenario. Some strategies may be more reliable "
            "but require more retries or virtual waiting."
        )

with detail:
    st.subheader("Saved measurements")
    st.caption(
        "Only allowlisted measurement fields are shown. "
        "Prompts, full responses and traces are excluded."
    )
    st.dataframe(
        [{name: row[name] for name in EXPORT_COLUMNS} for row in filtered],
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download filtered measurements (CSV)",
        data=export_csv(filtered).encode("utf-8-sig"),
        file_name="llm-token-economics-measurements.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "LLM Token Economics · Offline dashboard · Read-only · "
    "Descriptive statistics only"
)
