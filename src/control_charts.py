"""
control_charts.py
-------------------
Computes X-bar and R control charts and flags out-of-control signals.

QUALITY ENGINEERING BACKGROUND:

X-BAR CHART: tracks the subgroup MEAN over time -- answers "is the process
centered where it should be?"

R CHART: tracks the subgroup RANGE (max - min within each subgroup) over time --
answers "is the process's short-term variability staying consistent?" You need
BOTH charts together: a process can have a perfectly centered mean while its
spread is quietly growing out of control, or vice versa.

CONTROL LIMITS: Unlike the tolerance SPEC (which comes from engineering/customer
requirements -- "the part must be between 1.95 and 2.05mm"), control limits are
derived STATISTICALLY from the process's own natural behavior (typically mean +/-
3 standard deviations of the subgroup statistic). This is a critical distinction
in quality engineering: control limits tell you whether the process is behaving
CONSISTENTLY; the spec tells you whether the output is ACCEPTABLE. A process can
be "in control" (stable, predictable) while still producing parts outside of spec
if it's not CAPABLE enough -- that's what Cpk (next module) measures separately.

We use the standard control-chart constants (A2, D3, D4) for subgroup size n=5,
which come from statistical tables built into every quality engineering textbook
and used industry-wide (e.g., per ASQ / AIAG SPC reference manuals).

WESTERN ELECTRIC RULES: beyond just "did a point cross the control limit," real
quality systems also flag more subtle out-of-control PATTERNS, since a process
can be drifting before any single point crosses a hard limit. We implement two
of the most common:
  Rule 1: Any single point beyond 3-sigma control limits
  Rule 2: 8 consecutive points on the same side of the center line (indicates a
          sustained shift, exactly what "process drift" or a "sudden shift" event
          would produce)
"""

import numpy as np

# Standard SPC control chart constants for subgroup size n=5
# (from standard statistical process control reference tables)
A2_N5 = 0.577  # for X-bar chart control limits
D3_N5 = 0.0    # for R chart lower control limit
D4_N5 = 2.114  # for R chart upper control limit
D2_N5 = 2.326  # for estimating sigma from R-bar


def compute_control_charts(subgroups, baseline_subgroups=25):
    """
    Compute X-bar and R chart statistics and control limits.

    IMPORTANT SPC METHODOLOGY NOTE: control limits are established using only a
    BASELINE period of data assumed to represent stable, in-control production
    (here, the first `baseline_subgroups` subgroups) -- NOT the full dataset.
    This mirrors real practice: you characterize the process while it's known to
    be behaving normally, then hold those limits FIXED to monitor future
    production. If you instead computed limits from data that already contains
    a drift or shift, the out-of-control data would contaminate the baseline
    and mask the very problem you're trying to detect (a common real-world SPC
    mistake).

    Returns a dict with subgroup means, ranges, center lines, and control limits.
    """
    subgroup_means = subgroups.mean(axis=1)
    subgroup_ranges = subgroups.max(axis=1) - subgroups.min(axis=1)

    baseline_means = subgroup_means[:baseline_subgroups]
    baseline_ranges = subgroup_ranges[:baseline_subgroups]

    x_bar_bar = baseline_means.mean()  # baseline grand mean = center line of X-bar chart
    r_bar = baseline_ranges.mean()     # baseline average range = center line of R chart

    # X-bar chart control limits
    ucl_x = x_bar_bar + A2_N5 * r_bar
    lcl_x = x_bar_bar - A2_N5 * r_bar

    # R chart control limits
    ucl_r = D4_N5 * r_bar
    lcl_r = D3_N5 * r_bar

    # Estimate the process's natural (short-term) standard deviation from R-bar --
    # a standard SPC technique, more robust than a naive pooled std when the
    # process may already be drifting.
    sigma_estimate = r_bar / D2_N5

    return {
        "subgroup_means": subgroup_means,
        "subgroup_ranges": subgroup_ranges,
        "x_bar_bar": x_bar_bar,
        "r_bar": r_bar,
        "ucl_x": ucl_x,
        "lcl_x": lcl_x,
        "ucl_r": ucl_r,
        "lcl_r": lcl_r,
        "sigma_estimate": sigma_estimate,
    }


def detect_out_of_control(subgroup_means, center_line, ucl, lcl):
    """
    Apply simplified Western Electric rules to flag out-of-control subgroups.

    Rule 1: point beyond control limits
    Rule 2: 8 consecutive points on the same side of the center line

    Returns a boolean array (True = flagged) and a list of reasons.
    """
    n = len(subgroup_means)
    flagged = np.zeros(n, dtype=bool)
    reasons = ["" for _ in range(n)]

    # Rule 1: beyond control limits
    for i in range(n):
        if subgroup_means[i] > ucl or subgroup_means[i] < lcl:
            flagged[i] = True
            reasons[i] = "Rule 1: beyond control limits"

    # Rule 2: 8 consecutive points on the same side of center line
    side = np.sign(subgroup_means - center_line)
    run_length = 0
    run_side = 0
    for i in range(n):
        if side[i] == run_side and side[i] != 0:
            run_length += 1
        else:
            run_length = 1
            run_side = side[i]

        if run_length >= 8:
            flagged[i] = True
            if reasons[i] == "":
                reasons[i] = "Rule 2: 8+ consecutive points on one side of center"

    return flagged, reasons


if __name__ == "__main__":
    data = np.load("/home/claude/spc_project/data/process_data.npz")
    subgroups = data['subgroups']

    charts = compute_control_charts(subgroups)
    flagged, reasons = detect_out_of_control(
        charts['subgroup_means'], charts['x_bar_bar'], charts['ucl_x'], charts['lcl_x']
    )

    print(f"Grand mean (X-bar-bar): {charts['x_bar_bar']:.4f}mm")
    print(f"Average range (R-bar):  {charts['r_bar']:.4f}mm")
    print(f"X-bar chart limits:     LCL={charts['lcl_x']:.4f}  UCL={charts['ucl_x']:.4f}")
    print(f"R chart limits:         LCL={charts['lcl_r']:.4f}  UCL={charts['ucl_r']:.4f}")
    print(f"Estimated process sigma: {charts['sigma_estimate']:.4f}mm")
    print(f"\nOut-of-control subgroups flagged: {np.sum(flagged)} / {len(flagged)}")
    for i, (f, r) in enumerate(zip(flagged, reasons)):
        if f:
            print(f"  Subgroup {i}: {r} (mean={charts['subgroup_means'][i]:.4f}mm)")

    np.savez("/home/claude/spc_project/data/control_chart_results.npz",
             subgroup_means=charts['subgroup_means'],
             subgroup_ranges=charts['subgroup_ranges'],
             x_bar_bar=charts['x_bar_bar'], r_bar=charts['r_bar'],
             ucl_x=charts['ucl_x'], lcl_x=charts['lcl_x'],
             ucl_r=charts['ucl_r'], lcl_r=charts['lcl_r'],
             sigma_estimate=charts['sigma_estimate'],
             flagged=flagged)
    print("\nSaved control_chart_results.npz")
