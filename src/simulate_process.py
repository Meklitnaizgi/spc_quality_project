"""
simulate_process.py
---------------------
Simulates a medical device manufacturing measurement process: a critical dimension
(e.g., catheter tube outer diameter) being measured on samples pulled from
production over time.

QUALITY ENGINEERING BACKGROUND:

In real manufacturing, you don't measure every single part (too slow/expensive).
Instead, you pull small SUBGROUPS of parts at regular intervals (e.g., 5 parts
every hour) -- this is called "rational subgrouping." The idea: variation WITHIN
a subgroup captures normal short-term process noise, while variation BETWEEN
subgroup averages over time reveals whether the process itself is drifting.

We simulate a target dimension of 2.00mm (a plausible catheter OD) with a
tolerance spec of +/-0.05mm (i.e., acceptable range: 1.95mm - 2.05mm), and
inject two realistic manufacturing problems partway through the run:

1. GRADUAL DRIFT: simulates tool wear -- a mold or die slowly deforming over
   thousands of cycles, causing the mean dimension to creep away from target.
2. SUDDEN SHIFT: simulates a discrete event -- e.g., an operator changing a
   material lot, or a machine setting being bumped -- causing an abrupt jump
   in the process mean.

Both are textbook special-cause variation patterns that a Statistical Process
Control system is specifically designed to catch.
"""

import numpy as np


def simulate_manufacturing_process(
    n_subgroups=50,
    subgroup_size=5,
    target=2.00,
    process_std=0.012,
    drift_start_subgroup=30,
    drift_per_subgroup=0.0015,
    shift_subgroup=40,
    shift_amount=0.03,
    seed=11,
):
    """
    Simulate subgrouped measurement data from a manufacturing process.

    Returns
    -------
    subgroups : array of shape (n_subgroups, subgroup_size) -- raw measurements (mm)
    true_process_mean : array of shape (n_subgroups,) -- the TRUE underlying mean at
                         each subgroup (ground truth, for us to validate detection against)
    """
    rng = np.random.default_rng(seed)
    subgroups = np.zeros((n_subgroups, subgroup_size))
    true_process_mean = np.zeros(n_subgroups)

    for i in range(n_subgroups):
        mean_i = target

        # Gradual drift: tool wear accumulating after drift_start_subgroup
        if i >= drift_start_subgroup:
            mean_i += (i - drift_start_subgroup + 1) * drift_per_subgroup

        # Sudden shift: an abrupt jump starting at shift_subgroup
        if i >= shift_subgroup:
            mean_i += shift_amount

        true_process_mean[i] = mean_i
        subgroups[i, :] = rng.normal(mean_i, process_std, subgroup_size)

    return subgroups, true_process_mean


if __name__ == "__main__":
    subgroups, true_mean = simulate_manufacturing_process()

    print(f"Simulated {subgroups.shape[0]} subgroups of {subgroups.shape[1]} parts each "
          f"({subgroups.size} total measurements).")
    print(f"Target dimension: 2.00mm, Tolerance spec: +/-0.05mm (1.95-2.05mm)")
    print(f"Overall mean: {subgroups.mean():.4f}mm, Overall std: {subgroups.std():.4f}mm")

    np.savez("/home/claude/spc_project/data/process_data.npz",
             subgroups=subgroups, true_mean=true_mean)
    print("Saved to data/process_data.npz")
