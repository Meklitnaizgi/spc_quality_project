"""
gage_rr.py
-----------
Measurement System Analysis (MSA) via Gage R&R -- a core Lean Six Sigma "Measure
phase" tool, run BEFORE trusting any process data.

LEAN SIX SIGMA BACKGROUND:

Before analyzing a process, Six Sigma methodology requires validating the
MEASUREMENT SYSTEM itself. If your measurement system has too much variation
relative to the actual part-to-part variation, your control charts and Cpk
numbers are measuring noise from your gage/inspector, not the real process --
a classic root cause of chasing phantom process problems that don't actually exist.

Gage R&R decomposes total observed variation into three sources:
    Repeatability : variation when the SAME operator measures the SAME part
                    multiple times (equipment/gage variation)
    Reproducibility: variation BETWEEN DIFFERENT operators measuring the SAME parts
                    (operator-to-operator variation)
    Part-to-Part  : the REAL variation between different parts (what you actually
                    want to be measuring)

We simulate a standard Gage R&R study design: multiple operators, each measuring
the same set of parts multiple times, then use the ANOVA method (the more rigorous
of the two standard methods -- the other being the simpler "Range method") to
decompose variance.

%GRR (percent of total variation from the measurement system) is the key output:
    %GRR < 10%  : measurement system ACCEPTABLE
    %GRR 10-30% : measurement system MARGINAL, may be acceptable depending on application
    %GRR > 30%  : measurement system UNACCEPTABLE -- must be improved before trusting
                  any data collected with it
"""

import numpy as np


def simulate_gage_rr_study(n_parts=10, n_operators=3, n_trials=2,
                            part_std=0.015, repeatability_std=0.004,
                            reproducibility_std=0.003, seed=99):
    """
    Simulate a standard Gage R&R study: n_operators each measure n_parts,
    n_trials times each.

    Returns
    -------
    measurements : array of shape (n_parts, n_operators, n_trials)
    true_part_values : the TRUE underlying dimension of each part (ground truth,
                        not known to operators -- used only to validate our
                        variance decomposition)
    """
    rng = np.random.default_rng(seed)

    true_part_values = rng.normal(2.00, part_std, n_parts)
    # Each operator has a small consistent bias (reproducibility source)
    operator_bias = rng.normal(0, reproducibility_std, n_operators)

    measurements = np.zeros((n_parts, n_operators, n_trials))
    for p in range(n_parts):
        for o in range(n_operators):
            for t in range(n_trials):
                # Repeatability: random measurement noise each trial
                measurements[p, o, t] = (true_part_values[p] + operator_bias[o]
                                          + rng.normal(0, repeatability_std))

    return measurements, true_part_values


def anova_gage_rr(measurements):
    """
    ANOVA-method Gage R&R variance decomposition.

    measurements : array of shape (n_parts, n_operators, n_trials)
    """
    n_parts, n_operators, n_trials = measurements.shape
    n_total = n_parts * n_operators * n_trials

    grand_mean = measurements.mean()

    # Part means (averaged across operators and trials)
    part_means = measurements.mean(axis=(1, 2))
    # Operator means (averaged across parts and trials)
    operator_means = measurements.mean(axis=(0, 2))
    # Part x Operator cell means (averaged across trials)
    cell_means = measurements.mean(axis=2)

    # Sum of squares
    ss_part = n_operators * n_trials * np.sum((part_means - grand_mean) ** 2)
    ss_operator = n_parts * n_trials * np.sum((operator_means - grand_mean) ** 2)

    ss_interaction = 0.0
    for p in range(n_parts):
        for o in range(n_operators):
            predicted = part_means[p] + operator_means[o] - grand_mean
            ss_interaction += n_trials * (cell_means[p, o] - predicted) ** 2

    ss_repeatability = 0.0
    for p in range(n_parts):
        for o in range(n_operators):
            for t in range(n_trials):
                ss_repeatability += (measurements[p, o, t] - cell_means[p, o]) ** 2

    # Degrees of freedom
    df_part = n_parts - 1
    df_operator = n_operators - 1
    df_interaction = df_part * df_operator
    df_repeatability = n_parts * n_operators * (n_trials - 1)

    # Mean squares
    ms_part = ss_part / df_part
    ms_operator = ss_operator / df_operator if df_operator > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 1e-12
    ms_repeatability = ss_repeatability / df_repeatability

    # Variance component estimates (standard ANOVA Gage R&R formulas)
    var_repeatability = max(ms_repeatability, 0)
    var_interaction = max((ms_interaction - ms_repeatability) / n_trials, 0)
    var_operator = max((ms_operator - ms_interaction) / (n_parts * n_trials), 0)
    var_reproducibility = var_operator + var_interaction
    var_part = max((ms_part - ms_interaction) / (n_operators * n_trials), 0)

    var_grr = var_repeatability + var_reproducibility
    var_total = var_grr + var_part

    pct_grr = 100 * np.sqrt(var_grr / var_total) if var_total > 0 else 0
    pct_part = 100 * np.sqrt(var_part / var_total) if var_total > 0 else 0

    return {
        "var_repeatability": var_repeatability,
        "var_reproducibility": var_reproducibility,
        "var_grr": var_grr,
        "var_part": var_part,
        "var_total": var_total,
        "pct_grr": pct_grr,
        "pct_part": pct_part,
    }


def interpret_grr(pct_grr):
    if pct_grr < 10:
        return "ACCEPTABLE -- measurement system variation is a small fraction of total variation"
    elif pct_grr < 30:
        return "MARGINAL -- measurement system may need improvement depending on application criticality"
    else:
        return "UNACCEPTABLE -- measurement system variation is too large; process data cannot be trusted until this is fixed"


if __name__ == "__main__":
    measurements, true_values = simulate_gage_rr_study()
    n_parts, n_operators, n_trials = measurements.shape

    print("=" * 60)
    print("MEASUREMENT SYSTEM ANALYSIS -- Gage R&R Study")
    print("=" * 60)
    print(f"Study design: {n_parts} parts x {n_operators} operators x {n_trials} trials "
          f"= {measurements.size} total measurements")

    results = anova_gage_rr(measurements)

    print(f"\nVariance components:")
    print(f"  Repeatability (equipment):   {results['var_repeatability']:.8f}")
    print(f"  Reproducibility (operators): {results['var_reproducibility']:.8f}")
    print(f"  Total Gage R&R:              {results['var_grr']:.8f}")
    print(f"  Part-to-Part:                {results['var_part']:.8f}")

    print(f"\n%GRR (measurement system's share of total variation): {results['pct_grr']:.1f}%")
    print(f"%Part-to-Part (real process variation): {results['pct_part']:.1f}%")
    print(f"\nVerdict: {interpret_grr(results['pct_grr'])}")

    np.savez("/home/claude/spc_project/data/gage_rr_results.npz",
             measurements=measurements, true_values=true_values,
             pct_grr=results['pct_grr'], pct_part=results['pct_part'])
