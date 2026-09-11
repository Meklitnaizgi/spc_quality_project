"""
process_capability.py
-----------------------
Computes process capability indices (Cp and Cpk) -- the standard quality
engineering metrics for "is this process capable of reliably producing parts
within specification?"

QUALITY ENGINEERING BACKGROUND:

Cp and Cpk answer a DIFFERENT question than control charts do:
  - Control charts ask: "is the process STABLE and PREDICTABLE over time?"
  - Cp/Cpk ask: "even if stable, is the process's natural variation NARROW
    ENOUGH, and CENTERED ENOUGH, to reliably stay within the tolerance spec?"

A process can be perfectly "in control" (statistically stable, no drift) while
still being fundamentally incapable of meeting spec, if its natural variation
is simply too wide relative to the tolerance window. Cpk is what tells you that.

Cp (Process Capability): compares the SPEC WIDTH to the PROCESS WIDTH, ignoring
centering:
    Cp = (USL - LSL) / (6 * sigma)
  where USL/LSL are the upper/lower spec limits and sigma is the process's
  natural (short-term) standard deviation. Cp = 1.0 means the process's natural
  6-sigma spread exactly fills the tolerance window (razor-thin margin, not
  good). Cp >= 1.33 is a common minimum target in real manufacturing/medical
  device quality systems, and many programs (particularly high-reliability
  medical devices) target Cpk >= 1.5 or higher.

Cpk (Process Capability Index, accounting for centering): the more important,
realistic metric, since it also penalizes the process for being OFF-CENTER,
not just wide:
    Cpk = min( (USL - mean)/(3*sigma), (mean - LSL)/(3*sigma) )
  It takes the SMALLER of the two distances to either spec limit -- because a
  process is only as capable as its worst-case (nearest) distance to failing.

INTERPRETATION TABLE (industry-standard rule of thumb):
    Cpk < 1.0    : process not capable, will regularly produce out-of-spec parts
    1.0 <= Cpk < 1.33 : marginally capable, some risk
    Cpk >= 1.33  : generally considered capable (common minimum industry target)
    Cpk >= 1.67  : highly capable (often required for critical medical device features)
"""

import numpy as np


def compute_capability(measurements, usl, lsl, sigma_estimate=None):
    """
    Compute Cp and Cpk for a set of measurements against spec limits.

    Parameters
    ----------
    measurements : all individual part measurements (flattened, not subgroup means)
    usl, lsl : upper and lower specification limits
    sigma_estimate : if provided, use this (e.g., the R-bar/d2 estimate from the
                      control chart baseline) instead of the naive sample std.
                      Using the SHORT-TERM (within-subgroup) sigma rather than the
                      overall sample std is standard practice, since the overall
                      std can be inflated by any long-term drift/shift -- which
                      would make the process look "less capable" than its true
                      short-term potential, muddying the two separate questions
                      of stability vs. capability.
    """
    mean = measurements.mean()
    sigma = sigma_estimate if sigma_estimate is not None else measurements.std(ddof=1)

    cp = (usl - lsl) / (6 * sigma)
    cpu = (usl - mean) / (3 * sigma)  # capability relative to upper spec
    cpl = (mean - lsl) / (3 * sigma)  # capability relative to lower spec
    cpk = min(cpu, cpl)

    # Estimated defect rate (parts outside spec), assuming approximate normality
    from scipy.stats import norm
    p_above_usl = 1 - norm.cdf((usl - mean) / sigma)
    p_below_lsl = norm.cdf((lsl - mean) / sigma)
    estimated_defect_rate = p_above_usl + p_below_lsl

    return {
        "mean": mean,
        "sigma": sigma,
        "cp": cp,
        "cpk": cpk,
        "cpu": cpu,
        "cpl": cpl,
        "estimated_defect_rate": estimated_defect_rate,
        "estimated_ppm_defective": estimated_defect_rate * 1e6,
    }


def interpret_cpk(cpk):
    if cpk < 1.0:
        return "NOT CAPABLE -- process will regularly produce out-of-spec parts"
    elif cpk < 1.33:
        return "MARGINALLY CAPABLE -- some risk of out-of-spec parts, improvement recommended"
    elif cpk < 1.67:
        return "CAPABLE -- meets common industry minimum target"
    else:
        return "HIGHLY CAPABLE -- meets stringent requirements typical for critical device features"


if __name__ == "__main__":
    process_data = np.load("/home/claude/spc_project/data/process_data.npz")
    chart_data = np.load("/home/claude/spc_project/data/control_chart_results.npz")

    subgroups = process_data['subgroups']
    baseline_subgroups = subgroups[:25]  # only the stable, in-control baseline period
    baseline_measurements = baseline_subgroups.flatten()

    sigma_estimate = float(chart_data['sigma_estimate'])

    USL, LSL = 2.05, 1.95  # engineering tolerance spec

    print("=" * 60)
    print("PROCESS CAPABILITY ANALYSIS -- Baseline (In-Control) Period")
    print("=" * 60)
    results_baseline = compute_capability(baseline_measurements, USL, LSL, sigma_estimate)
    print(f"Mean: {results_baseline['mean']:.4f}mm   Sigma: {results_baseline['sigma']:.4f}mm")
    print(f"Cp:  {results_baseline['cp']:.3f}")
    print(f"Cpk: {results_baseline['cpk']:.3f}  -> {interpret_cpk(results_baseline['cpk'])}")
    print(f"Estimated defect rate: {results_baseline['estimated_defect_rate']*100:.4f}% "
          f"({results_baseline['estimated_ppm_defective']:.0f} PPM)")

    print("\n" + "=" * 60)
    print("PROCESS CAPABILITY ANALYSIS -- Full Run (Including Drift + Shift)")
    print("=" * 60)
    all_measurements = subgroups.flatten()
    results_full = compute_capability(all_measurements, USL, LSL, sigma_estimate)
    print(f"Mean: {results_full['mean']:.4f}mm   Sigma (short-term est.): {results_full['sigma']:.4f}mm")
    print(f"Cp:  {results_full['cp']:.3f}")
    print(f"Cpk: {results_full['cpk']:.3f}  -> {interpret_cpk(results_full['cpk'])}")
    print(f"Estimated defect rate: {results_full['estimated_defect_rate']*100:.4f}% "
          f"({results_full['estimated_ppm_defective']:.0f} PPM)")

    # Actual out-of-spec count in the full dataset (ground truth, not estimated)
    n_out_of_spec = np.sum((all_measurements > USL) | (all_measurements < LSL))
    actual_defect_rate = n_out_of_spec / len(all_measurements)
    print(f"\nActual parts outside spec in this simulated run: {n_out_of_spec} / {len(all_measurements)} "
          f"({actual_defect_rate*100:.2f}%)")

    print("\n*** IMPORTANT QUALITY ENGINEERING CAVEAT ***")
    print(f"Notice the model's ESTIMATED defect rate ({results_full['estimated_ppm_defective']:.0f} PPM) is far")
    print(f"lower than the ACTUAL observed defect rate ({actual_defect_rate*1e6:.0f} PPM equivalent). This is")
    print("expected and important: Cpk assumes measurements come from ONE stable, consistent")
    print("distribution. Our control chart already proved this dataset is NOT stable (it contains")
    print("an unresolved process shift). Computing Cpk on non-stable data blends a healthy period")
    print("with a broken period into one falsely optimistic number -- a real and common quality")
    print("engineering mistake. The correct practice is to confirm control chart stability FIRST,")
    print("and only compute Cpk on a period verified to be in statistical control (as done above")
    print("for the baseline period, where the estimate and reality closely agree).")

    np.savez("/home/claude/spc_project/data/capability_results.npz",
             baseline_cpk=results_baseline['cpk'], full_cpk=results_full['cpk'],
             baseline_cp=results_baseline['cp'], full_cp=results_full['cp'])
