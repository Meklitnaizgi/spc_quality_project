"""
doe_experiment.py
-------------------
Design of Experiments (DOE): a 2^2 full factorial experiment identifying which
process inputs significantly affect the output dimension, and whether they
interact.

LEAN SIX SIGMA BACKGROUND:

Control charts and Cpk are PASSIVE tools -- they monitor a process and tell you
IF something is wrong, but not WHY. DOE is an ACTIVE tool used in the
Analyze/Improve phases of DMAIC: deliberately and systematically vary suspected
process inputs (called "factors") across a planned set of combinations, measure
the output, and use statistics to determine which factors actually matter.

We simulate a 2^2 FULL FACTORIAL design: 2 factors, each tested at 2 levels
(low/high), run across all 4 combinations, with replicates for statistical power.
This is the simplest true factorial design and the standard entry point taught
in Lean Six Sigma Green Belt / Black Belt curricula.

Factors modeled here (plausible for an extrusion/molding process producing a
tube-like medical device component):
    Factor A: Extrusion Temperature (Low / High)
    Factor B: Line Speed (Low / High)

We estimate:
    MAIN EFFECTS: how much the output changes, on average, when a single factor
                  moves from low to high (holding the other factor's average level)
    INTERACTION EFFECT: whether the effect of one factor DEPENDS on the level of
                  the other factor -- a critical concept DOE is specifically
                  designed to detect, which simpler "change one factor at a time"
                  experimentation misses entirely.

We use a simple, transparent effects-estimation approach (Yates-style contrast
calculation) plus a two-way ANOVA-style significance check, rather than
importing a full DOE library, so every step is visible and explainable.
"""

import numpy as np
from scipy import stats


def simulate_doe_experiment(true_temp_effect=0.020, true_speed_effect=-0.008,
                             true_interaction_effect=0.015, noise_std=0.006,
                             n_replicates=4, seed=21):
    """
    Simulate a 2^2 full factorial DOE run.

    Coded factor levels: -1 = Low, +1 = High (standard DOE coding convention)

    Returns a structured array of results: one row per run, columns
    [temp_level, speed_level, replicate, measured_output]
    """
    rng = np.random.default_rng(seed)
    baseline = 2.00

    levels = [-1, 1]
    rows = []
    for temp in levels:
        for speed in levels:
            true_mean = (baseline
                         + (true_temp_effect / 2) * temp
                         + (true_speed_effect / 2) * speed
                         + (true_interaction_effect / 2) * temp * speed)
            for r in range(n_replicates):
                measured = true_mean + rng.normal(0, noise_std)
                rows.append([temp, speed, r, measured])

    return np.array(rows)


def analyze_doe(doe_data):
    """
    Estimate main effects and interaction effect from 2^2 factorial data, plus
    run a two-way ANOVA to assess statistical significance.
    """
    temp = doe_data[:, 0]
    speed = doe_data[:, 1]
    y = doe_data[:, 3]

    # Effect estimates: standard 2^2 factorial contrast method.
    # The "effect" of a factor = average response at High level minus average
    # response at Low level.
    mean_temp_high = y[temp == 1].mean()
    mean_temp_low = y[temp == -1].mean()
    effect_temp = mean_temp_high - mean_temp_low

    mean_speed_high = y[speed == 1].mean()
    mean_speed_low = y[speed == -1].mean()
    effect_speed = mean_speed_high - mean_speed_low

    # Interaction effect: difference in temp's effect depending on speed level
    mean_pp = y[(temp == 1) & (speed == 1)].mean()
    mean_pm = y[(temp == 1) & (speed == -1)].mean()
    mean_mp = y[(temp == -1) & (speed == 1)].mean()
    mean_mm = y[(temp == -1) & (speed == -1)].mean()
    effect_interaction = (mean_pp - mean_pm - mean_mp + mean_mm) / 1  # standard contrast

    # Two-way ANOVA via statsmodels-style manual computation (kept dependency-light)
    groups = {}
    for t in [-1, 1]:
        for s in [-1, 1]:
            groups[(t, s)] = y[(temp == t) & (speed == s)]

    grand_mean = y.mean()
    n_per_cell = len(groups[(-1, -1)])

    ss_temp = 2 * n_per_cell * ((mean_temp_high - grand_mean) ** 2 + (mean_temp_low - grand_mean) ** 2) / 2 * 2
    # (Recompute cleanly using standard factorial ANOVA sums of squares)
    ss_temp = n_per_cell * 2 * np.sum([(y[temp == lvl].mean() - grand_mean) ** 2 for lvl in [-1, 1]])
    ss_speed = n_per_cell * 2 * np.sum([(y[speed == lvl].mean() - grand_mean) ** 2 for lvl in [-1, 1]])

    cell_means = {k: v.mean() for k, v in groups.items()}
    ss_interaction = n_per_cell * sum(
        (cell_means[(t, s)] - (y[temp == t].mean() + y[speed == s].mean() - grand_mean)) ** 2
        for t in [-1, 1] for s in [-1, 1]
    )

    ss_error = sum(np.sum((v - v.mean()) ** 2) for v in groups.values())
    ss_total = np.sum((y - grand_mean) ** 2)

    df_error = len(y) - 4  # 4 cell means estimated
    ms_error = ss_error / df_error

    f_temp = (ss_temp / 1) / ms_error
    f_speed = (ss_speed / 1) / ms_error
    f_interaction = (ss_interaction / 1) / ms_error

    p_temp = 1 - stats.f.cdf(f_temp, 1, df_error)
    p_speed = 1 - stats.f.cdf(f_speed, 1, df_error)
    p_interaction = 1 - stats.f.cdf(f_interaction, 1, df_error)

    return {
        "effect_temp": effect_temp,
        "effect_speed": effect_speed,
        "effect_interaction": effect_interaction,
        "p_temp": p_temp,
        "p_speed": p_speed,
        "p_interaction": p_interaction,
        "cell_means": cell_means,
    }


if __name__ == "__main__":
    doe_data = simulate_doe_experiment()
    n_runs = doe_data.shape[0]
    print("=" * 60)
    print("DESIGN OF EXPERIMENTS (DOE) -- 2^2 Full Factorial Study")
    print("=" * 60)
    print(f"Factors: A = Extrusion Temperature, B = Line Speed")
    print(f"Design: 2^2 factorial, {n_runs} total runs ({n_runs//4} replicates per combination)\n")

    results = analyze_doe(doe_data)

    alpha = 0.05
    for name, effect, p in [("Temperature (A)", results['effect_temp'], results['p_temp']),
                              ("Speed (B)", results['effect_speed'], results['p_speed']),
                              ("Temperature x Speed interaction (AB)",
                               results['effect_interaction'], results['p_interaction'])]:
        sig = "SIGNIFICANT" if p < alpha else "not significant"
        print(f"{name}:")
        print(f"  Estimated effect: {effect:+.4f}mm   p-value: {p:.4f}   -> {sig} (alpha=0.05)")

    print("\nInterpretation:")
    print(f"  Moving Temperature from Low to High changes the dimension by "
          f"{results['effect_temp']:+.4f}mm on average.")
    print(f"  Moving Speed from Low to High changes the dimension by "
          f"{results['effect_speed']:+.4f}mm on average.")
    if results['p_interaction'] < alpha:
        print(f"  The SIGNIFICANT interaction effect means Temperature's impact DEPENDS on")
        print(f"  the Speed setting -- adjusting Temperature alone without considering Speed")
        print(f"  would give inconsistent, confusing results on the production floor. This is")
        print(f"  exactly the kind of finding a simple 'change one factor at a time' approach")
        print(f"  (instead of a proper factorial DOE) would have completely missed.")

    np.savez("/home/claude/spc_project/data/doe_results.npz",
             doe_data=doe_data, effect_temp=results['effect_temp'],
             effect_speed=results['effect_speed'],
             effect_interaction=results['effect_interaction'])
