"""
improvement_validation.py
---------------------------
Hypothesis testing to statistically validate a process improvement -- the
"Improve" phase of DMAIC, and the "Control" phase's evidence that a fix actually
worked, not just a random good week.

LEAN SIX SIGMA BACKGROUND:

After identifying a root cause (via control charts + DOE above) and implementing
a fix, Six Sigma methodology requires PROVING the fix worked with statistics --
not just eyeballing "it looks better now." A two-sample hypothesis test (Welch's
t-test, which doesn't assume equal variances between groups) answers: "is the
difference between the old process and the new process larger than what we'd
expect from random chance alone?"

We simulate implementing a corrective action informed by our DOE findings above
(e.g., tightening temperature control based on the significant interaction
effect we found) and test whether the resulting process shows a statistically
significant reduction in variability (not just a different mean).
"""

import numpy as np
from scipy import stats


def simulate_improved_process(n_subgroups=20, subgroup_size=5, target=2.00,
                                improved_std=0.007, seed=55):
    """
    Simulate process data AFTER a corrective action -- tighter process control
    (lower standard deviation) informed by the DOE findings.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(target, improved_std, (n_subgroups, subgroup_size))


def compare_variance(before_data, after_data, alpha=0.05):
    """
    Levene's test for equality of variances -- more robust to non-normality than
    the classic F-test, standard practice for this kind of before/after
    manufacturing comparison.
    """
    stat, p_value = stats.levene(before_data.flatten(), after_data.flatten())
    significant = p_value < alpha
    return {"statistic": stat, "p_value": p_value, "significant": significant}


def compare_mean(before_data, after_data, alpha=0.05):
    """Welch's t-test for difference in means (does not assume equal variances)."""
    stat, p_value = stats.ttest_ind(before_data.flatten(), after_data.flatten(), equal_var=False)
    significant = p_value < alpha
    return {"statistic": stat, "p_value": p_value, "significant": significant}


if __name__ == "__main__":
    process_data = np.load("/home/claude/spc_project/data/process_data.npz")
    before = process_data['subgroups'][:25]  # baseline (in-control, before any known problem)

    after = simulate_improved_process()

    print("=" * 60)
    print("IMPROVEMENT VALIDATION -- Before vs. After Corrective Action")
    print("=" * 60)
    print(f"Before: mean={before.mean():.4f}mm, std={before.std(ddof=1):.4f}mm, n={before.size}")
    print(f"After:  mean={after.mean():.4f}mm, std={after.std(ddof=1):.4f}mm, n={after.size}")

    var_test = compare_variance(before, after)
    print(f"\nLevene's test for equal variance:")
    print(f"  statistic={var_test['statistic']:.3f}, p-value={var_test['p_value']:.5f}")
    print(f"  -> {'SIGNIFICANT reduction in variability' if var_test['significant'] else 'No significant change in variability'} (alpha=0.05)")

    mean_test = compare_mean(before, after)
    print(f"\nWelch's t-test for equal mean:")
    print(f"  statistic={mean_test['statistic']:.3f}, p-value={mean_test['p_value']:.5f}")
    print(f"  -> {'SIGNIFICANT change in mean' if mean_test['significant'] else 'No significant change in mean (process remains correctly centered)'} (alpha=0.05)")

    if var_test['significant'] and not mean_test['significant']:
        print("\nConclusion: the corrective action significantly TIGHTENED the process (lower")
        print("variability) WITHOUT shifting it off-target -- exactly the desired outcome of a")
        print("well-executed Six Sigma improvement, and statistically defensible evidence for a")
        print("Control phase sign-off, not just a subjective impression that things improved.")
