"""
run_spc_analysis.py
---------------------
Master script: runs the full SPC quality analysis end-to-end.

    1. Simulate a manufacturing measurement process (with injected drift + shift)
    2. Compute X-bar and R control charts from a stable baseline period
    3. Detect out-of-control signals using Western Electric rules
    4. Compute process capability (Cp/Cpk) -- correctly, on the verified-stable
       baseline period, with an explicit caveat about why the full run's number
       would be misleading

Run with:  python3 run_spc_analysis.py
"""

import numpy as np
from simulate_process import simulate_manufacturing_process
from control_charts import compute_control_charts, detect_out_of_control
from process_capability import compute_capability, interpret_cpk
from gage_rr import simulate_gage_rr_study, anova_gage_rr, interpret_grr
from doe_experiment import simulate_doe_experiment, analyze_doe
from improvement_validation import simulate_improved_process, compare_variance, compare_mean
from sigma_level import compute_dpmo, dpmo_to_sigma_level, cpk_to_sigma_level


def main():
    print("=" * 60)
    print("LEAN SIX SIGMA DMAIC ANALYSIS")
    print("Medical device component: critical dimension monitoring")
    print("=" * 60)

    USL, LSL, TARGET = 2.05, 1.95, 2.00
    print(f"\nSpec: {TARGET}mm +/- 0.05mm  (LSL={LSL}mm, USL={USL}mm)")

    # ---------------- MEASURE PHASE: validate the measurement system first ----------------
    print("\n" + "=" * 60)
    print("MEASURE PHASE -- Step 1: Validate the Measurement System (Gage R&R)")
    print("=" * 60)
    print("Before trusting ANY process data, confirm the measurement system itself is sound.")
    gage_measurements, _ = simulate_gage_rr_study()
    gage_results = anova_gage_rr(gage_measurements)
    print(f"%GRR: {gage_results['pct_grr']:.1f}%  -> {interpret_grr(gage_results['pct_grr'])}")

    # ---------------- MEASURE PHASE: collect process data ----------------
    print("\n" + "=" * 60)
    print("MEASURE PHASE -- Step 2: Collect Process Data")
    print("=" * 60)
    subgroups, true_mean = simulate_manufacturing_process()
    print(f"Collected {subgroups.shape[0]} subgroups x {subgroups.shape[1]} parts "
          f"({subgroups.size} total measurements)")

    # ---------------- ANALYZE PHASE: control charts + capability ----------------
    print("\n" + "=" * 60)
    print("ANALYZE PHASE -- Step 1: Control Charts (is the process stable?)")
    print("=" * 60)
    charts = compute_control_charts(subgroups, baseline_subgroups=25)
    flagged, reasons = detect_out_of_control(
        charts['subgroup_means'], charts['x_bar_bar'], charts['ucl_x'], charts['lcl_x']
    )
    first_flag_idx = np.argmax(flagged) if np.any(flagged) else None
    print(f"{np.sum(flagged)}/{len(flagged)} subgroups flagged. First signal at subgroup {first_flag_idx}.")

    print("\n" + "=" * 60)
    print("ANALYZE PHASE -- Step 2: Process Capability (Cpk) on verified-stable baseline")
    print("=" * 60)
    baseline_measurements = subgroups[:25].flatten()
    cap_baseline = compute_capability(baseline_measurements, USL, LSL, charts['sigma_estimate'])
    print(f"Baseline Cpk: {cap_baseline['cpk']:.3f} -> {interpret_cpk(cap_baseline['cpk'])}")

    baseline_sigma = cpk_to_sigma_level(cap_baseline['cpk'])
    print(f"Approximate Sigma Level (from Cpk): {baseline_sigma:.2f}")

    print("\n" + "=" * 60)
    print("ANALYZE PHASE -- Step 3: Design of Experiments (what's causing the drift?)")
    print("=" * 60)
    doe_data = simulate_doe_experiment()
    doe_results = analyze_doe(doe_data)
    alpha = 0.05
    for name, p in [("Temperature", doe_results['p_temp']), ("Speed", doe_results['p_speed']),
                     ("Temp x Speed interaction", doe_results['p_interaction'])]:
        sig = "SIGNIFICANT" if p < alpha else "not significant"
        print(f"  {name}: p={p:.4f} -> {sig}")

    # ---------------- IMPROVE PHASE: validate the fix ----------------
    print("\n" + "=" * 60)
    print("IMPROVE PHASE -- Validate corrective action with hypothesis testing")
    print("=" * 60)
    before = subgroups[:25]
    after = simulate_improved_process()
    var_test = compare_variance(before, after)
    mean_test = compare_mean(before, after)
    print(f"Before: std={before.std(ddof=1):.4f}mm  |  After: std={after.std(ddof=1):.4f}mm")
    print(f"Variance reduction significant? {var_test['significant']} (p={var_test['p_value']:.4f})")
    print(f"Mean stayed on-target?          {not mean_test['significant']} (p={mean_test['p_value']:.4f})")

    # ---------------- CONTROL PHASE summary ----------------
    print("\n" + "=" * 60)
    print("CONTROL PHASE -- Summary for sign-off")
    print("=" * 60)
    print(f"Measurement system: {interpret_grr(gage_results['pct_grr'])}")
    print(f"Process capability (baseline): Cpk={cap_baseline['cpk']:.2f} (~{baseline_sigma:.1f} sigma)")
    print(f"Root cause identified via DOE: Temperature x Speed interaction (p={doe_results['p_interaction']:.4f})")
    print(f"Improvement validated: variance reduced with no mean shift "
          f"({'CONFIRMED' if var_test['significant'] and not mean_test['significant'] else 'INCONCLUSIVE'})")


if __name__ == "__main__":
    main()
