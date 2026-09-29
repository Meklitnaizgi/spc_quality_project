# Lean Six Sigma DMAIC Analysis: Medical Device Manufacturing Quality

A full DMAIC (Define-Measure-Analyze-Improve-Control) quality engineering project
simulating a medical device manufacturing process (modeled on a catheter tube
outer diameter), applying the core Lean Six Sigma toolkit: Measurement System
Analysis, Statistical Process Control, Process Capability, Design of Experiments,
and hypothesis testing.


## Define

**Problem:** A critical dimension on a manufactured medical device component
(target 2.00mm, tolerance ±0.05mm) needs to stay reliably within specification.
The goal of this project: build and demonstrate the analytical toolkit to (1)
confirm the measurement system can be trusted, (2) monitor whether the process is
stable, (3) quantify whether it's capable of meeting spec, (4) identify root
causes when it isn't, and (5) statistically prove a fix actually worked.

## Measure

### Step 1 — Measurement System Analysis (Gage R&R)

Before trusting any process data, validated the measurement system itself using
an ANOVA-method Gage R&R study (10 parts × 3 operators × 2 trials = 60 measurements).

| Result | Value |
|---|---|
| %GRR (measurement system's share of total variation) | 23.4% |
| Verdict | **Marginal** — usable, but improvement recommended before high-stakes decisions |

This intentionally landed in a realistic "marginal" zone rather than a clean
pass — a common real-world outcome, and the correct conclusion is nuanced (not a
simple pass/fail), which is itself an important Six Sigma lesson.

### Step 2 — Data Collection

Simulated 50 subgroups of 5 parts each (250 measurements) sampled over production
time, with two realistic injected manufacturing problems (gradual tool-wear drift
starting at subgroup 30, and a sudden process shift at subgroup 40).

## Analyze

### Step 1 — Control Charts (is the process stable?)

X-bar and R control charts, with limits established from a verified-stable
baseline period (first 25 subgroups) — not contaminated by the later drift/shift.

| Metric | Result |
|---|---|
| False alarms during baseline | 0 / 25 |
| True drift injected at subgroup | 30 |
| First out-of-control signal detected at | 31 |
| True shift injected at subgroup | 40 |
| Shift detected at | 40 (immediate) |

### Step 2 — Process Capability (Cpk) and Sigma Level

| Period | Cpk | Approx. Sigma Level | Interpretation |
|---|---|---|---|
| Baseline (verified stable) | 1.52 | ~4.6σ | Capable, exceeds common 1.33 minimum |
| Full run (includes drift+shift) | 1.15 (misleading, see caveat) | — | — |

**Key caveat:** Cpk computed on the full run is misleading — it estimates ~273
defective PPM while the data actually contains 13.2% out-of-spec parts. This is
because Cpk assumes one stable distribution, and the control chart already proved
this dataset isn't stable. The capability histogram shows the full-run data is
visibly **bimodal**, not a single bell curve. Correct practice (applied here):
confirm stability first via control charts, only compute Cpk on a verified-stable
period.

### Step 3 — Design of Experiments (what's causing it?)

Ran a 2² full factorial DOE (Temperature × Line Speed, 4 replicates per
combination, 16 runs) to identify root causes rather than guessing.

| Factor | Effect | p-value | Significant? |
|---|---|---|---|
| Temperature | +0.0215mm | 0.0001 | Yes |
| Speed | -0.0133mm | 0.0032 | Yes |
| Temperature × Speed interaction | +0.0329mm | 0.0007 | **Yes** |

The significant interaction effect is the key finding: Temperature's impact on
the dimension depends on the Speed setting. A simpler "change one factor at a
time" approach (rather than a proper factorial design) would have completely
missed this — adjusting Temperature alone, without considering Speed, would
produce inconsistent results on the production floor.

## Improve

Simulated a corrective action (tighter process control informed by the DOE
findings) and statistically validated it with hypothesis testing rather than
relying on a subjective "looks better now."

| Test | Result | Conclusion |
|---|---|---|
| Levene's test (variance) | p = 0.0054 | Significant variance **reduction** |
| Welch's t-test (mean) | p = 0.6014 | No significant mean shift (still on-target) |

**Conclusion:** the corrective action significantly tightened the process
(reduced variability) without shifting it off-target — the ideal outcome of a
well-executed improvement, backed by statistical evidence suitable for a real
Control-phase sign-off.

## Control

Summary for process sign-off:
- Measurement system: Marginal (23.4% GRR) — recommend improvement before relying
  on this gage for tight-tolerance decisions
- Process capability: Cpk = 1.52 (~4.6σ) on verified-stable baseline
- Root cause of drift: Temperature × Speed interaction (DOE-confirmed, p=0.0007)
- Improvement: statistically confirmed variance reduction with no mean shift

## Technical Skills Demonstrated

- **Measurement System Analysis:** ANOVA-method Gage R&R, variance component
  decomposition, %GRR interpretation
- **Statistical Process Control:** X-bar/R control charts, rational subgrouping,
  Western Electric out-of-control rules, baseline-period methodology
- **Process Capability:** Cp/Cpk calculation, correct vs. misleading application,
  DPMO and Sigma Level conversion
- **Design of Experiments:** 2² full factorial design, main effects and
  interaction effect estimation, two-way ANOVA significance testing
- **Hypothesis testing:** Levene's test, Welch's t-test, before/after improvement
  validation
- **DMAIC methodology:** structuring an end-to-end quality investigation the way
  Lean Six Sigma projects are run in industry

## ⚠️ Limitations

- All data is simulated, not from a real production line or real gage study
- The DOE model assumes a simple linear + interaction effect structure; real
  processes can have more complex (e.g., curvature/quadratic) relationships that
  would require a more advanced design (e.g., central composite design)
- Gage R&R here uses a simplified ANOVA calculation; commercial statistical
  software (Minitab, JMP) includes additional refinements (e.g., negative
  variance component handling nuances) not fully replicated here

## Project Structure

```
spc_project/
├── src/
│   ├── simulate_process.py         # Manufacturing process simulator (with injected faults)
│   ├── control_charts.py           # X-bar/R chart computation + out-of-control detection
│   ├── process_capability.py       # Cp/Cpk analysis + capability interpretation
│   ├── sigma_level.py              # DPMO and Sigma Level calculation
│   ├── gage_rr.py                  # Measurement System Analysis (Gage R&R)
│   ├── doe_experiment.py           # Design of Experiments (2^2 factorial)
│   ├── improvement_validation.py   # Hypothesis testing for process improvement
│   └── run_spc_analysis.py         # End-to-end DMAIC orchestration script
├── data/
├── plots/
└── README.md
```

## Running It

```bash
cd src
python3 run_spc_analysis.py
```

## Possible Future Extensions

- Add a central composite / response surface design to model curvature effects
- Add a formal control plan document (reaction plans for each out-of-control rule)
- Model attribute (pass/fail) data with p-charts, alongside the variable data
  (X-bar/R) approach used here
- Extend Gage R&R to include a destructive-testing variant (relevant for many
  medical device test methods where the same part can't be re-measured)
