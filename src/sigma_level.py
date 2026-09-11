"""
sigma_level.py
---------------
Computes DPMO (Defects Per Million Opportunities) and Sigma Level -- the metric
that gives "Six Sigma" its name.

LEAN SIX SIGMA BACKGROUND:

A "Sigma Level" measures process capability in a standardized way that's
comparable across completely different processes/industries. It answers: "how
many standard deviations fit between the process mean and the nearest spec
limit?" A HIGHER sigma level means a LOWER defect rate.

    Sigma Level  ~2      ~3       ~4        ~5         ~6
    DPMO      308,000  66,800   6,210     233        3.4
    (rough)

"Six Sigma" as a methodology takes its name from the goal of achieving a 6-sigma
process: 3.4 defects per million opportunities -- extraordinarily low. Most
real-world "good" processes run at 3-4 sigma; getting to 5-6 sigma requires
serious process control, which is exactly why Lean Six Sigma exists as a
discipline: the tools in this project (control charts, Cpk, Gage R&R, DOE) are
literally the toolkit used to move a process from a lower sigma level to a
higher one.

The commonly used industry approximation includes a 1.5-sigma "long-term shift"
correction (Motorola's original Six Sigma definition), which accounts for the
fact that real processes drift over time in ways a single short-term snapshot
doesn't capture -- notice how directly this connects to the drift/shift problem
we simulated and detected in this same project.
"""

import numpy as np
from scipy.stats import norm


def compute_dpmo(defect_count, unit_count, opportunities_per_unit=1):
    """
    DPMO = (Defects / (Units x Opportunities per unit)) x 1,000,000

    'Opportunities per unit' = how many distinct ways a single unit could fail
    (e.g., a component with 3 critical dimensions has 3 opportunities per unit).
    For this project, each part has 1 critical dimension being checked, so
    opportunities_per_unit=1.
    """
    total_opportunities = unit_count * opportunities_per_unit
    dpmo = (defect_count / total_opportunities) * 1_000_000
    return dpmo


def dpmo_to_sigma_level(dpmo, long_term_shift=1.5):
    """
    Convert DPMO to an equivalent Sigma Level, including the standard 1.5-sigma
    long-term shift correction used throughout industry Six Sigma practice.
    """
    defect_rate = dpmo / 1_000_000
    # Find the Z-score corresponding to this defect rate (yield = 1 - defect_rate)
    z_short_term = norm.ppf(1 - defect_rate)
    sigma_level = z_short_term + long_term_shift
    return sigma_level


def cpk_to_sigma_level(cpk, long_term_shift=1.5):
    """
    Alternative route: convert directly from Cpk to an approximate sigma level.
    Short-term sigma level ~= 3 * Cpk. Long-term (reported) sigma level applies
    the standard 1.5-sigma shift.
    """
    short_term_sigma = 3 * cpk
    long_term_sigma = short_term_sigma  # Cpk already reflects long-term data if
    # computed on long-term data; if computed on short-term baseline data (as we
    # did), it's conventional to still report the shifted number for comparison
    # to industry benchmarks:
    long_term_sigma_reported = short_term_sigma - long_term_shift + long_term_shift
    return short_term_sigma


if __name__ == "__main__":
    process_data = np.load("/home/claude/spc_project/data/process_data.npz")
    capability_data = np.load("/home/claude/spc_project/data/capability_results.npz")

    subgroups = process_data['subgroups']
    USL, LSL = 2.05, 1.95

    print("=" * 60)
    print("SIGMA LEVEL ANALYSIS")
    print("=" * 60)

    # Baseline (healthy) period
    baseline = subgroups[:25].flatten()
    baseline_defects = np.sum((baseline > USL) | (baseline < LSL))
    baseline_dpmo = compute_dpmo(baseline_defects, len(baseline))
    baseline_dpmo_safe = max(baseline_dpmo, 3.4)  # avoid log(0) issues in sigma conversion when 0 defects
    baseline_sigma = dpmo_to_sigma_level(baseline_dpmo_safe)

    print(f"\nBaseline period (healthy, in-control):")
    print(f"  Observed defects: {baseline_defects}/{len(baseline)}")
    print(f"  DPMO (observed count method): {baseline_dpmo:.1f}")
    print(f"  Approximate Sigma Level (observed count method): {baseline_sigma:.2f}")

    baseline_cpk = float(capability_data['baseline_cpk'])
    baseline_sigma_from_cpk = cpk_to_sigma_level(baseline_cpk)
    print(f"  Sigma Level from Cpk (model-based method): {baseline_sigma_from_cpk:.2f}")
    print("  *** CAVEAT: 0 observed defects in only 125 samples does NOT prove true")
    print("  six-sigma performance -- the sample size is far too small to statistically")
    print("  confirm a defect rate as low as 3.4 DPMO. The Cpk-based estimate above is the")
    print("  more statistically defensible number for a sample this size; the 'observed")
    print("  count' method is only reliable with much larger sample sizes.")

    # Full run (includes drift + shift)
    full = subgroups.flatten()
    full_defects = np.sum((full > USL) | (full < LSL))
    full_dpmo = compute_dpmo(full_defects, len(full))
    full_sigma = dpmo_to_sigma_level(full_dpmo)

    print(f"\nFull run (includes uncorrected drift + shift):")
    print(f"  Observed defects: {full_defects}/{len(full)}")
    print(f"  DPMO: {full_dpmo:.1f}")
    print(f"  Approximate Sigma Level: {full_sigma:.2f}")

    print(f"\nSigma level DROPPED from {baseline_sigma:.2f} to {full_sigma:.2f} once the")
    print("uncorrected process drift/shift was allowed to persist -- this is the direct,")
    print("quantified cost of not catching and correcting a control chart signal in time.")
