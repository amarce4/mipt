"""Reusable analysis and plotting tools for the MIPT simulation outputs.

Public functions
----------------
gmn(...)               Plot GMN/fGMN and optional minimum bipartite negativity.
entropy(...)           Plot entropy scaling curves and weighted fits; ``order=1``
                       selects von Neumann, ``order=2`` (default) Rényi-2.
dist_scaling(...)      Plot two-site (MI/negativity) or three-site (TMI/GMN)
                       distance scaling and power-law fits as separate flush
                       ordinary/fermionic stacks, plus finite-size exponent
                       convergence; ``ent_decomp=True``
                       splits the pair negativities into the entangled fraction
                       and the conditional magnitude instead, and
                       ``paper_fit_ranges=True`` reproduces the published
                       two-party fit windows. Reads the aggregated CSV, the
                       legacy row-level CSV, the per-record binary, or the
                       summary CSV ``./summarize_records`` reduces one to;
                       ``show_dist=(L, d_min)`` adds a stacked-histogram figure
                       of the non-zero MI/fMI/GMN/fGMN distributions, which
                       needs the records.
dist_percolation(...) Joint spacetime-connectivity / entanglement diagnostics
                       from format v2 per-record binaries: endpoint survival
                       against q^2, P_perc, and the two conditionals
                       kappa = P(entangled | connected) and
                       eta = P(entangled | disconnected), swept over the
                       positivity threshold. Also fits P_inf(p) = A(1-p)^beta.
dist_pair_gap(...)    The connected-zero triple analysis: for graph-connected pairs
                       with no fermionic negativity, the fraction R_3(d) with an
                       fGMN-positive third site (trajectory-clustered errors),
                       the partition of P(C) - P(E_2), and a null comparison
                       against distance-matched disconnected pairs.
gate_path_gap_test(...) Gate-path diagnostic (MIPT_DIST_GATE_DIAGNOSTICS=1): streams
                       a record binary, matches every connected-zero pair to an
                       entangled connected pair from the same trajectory, and
                       compares widest/shortest-maximin gate entangling power
                       and parity-block mixing with trajectory-clustered CIs;
                       writes <stem>_gate_path_summary.csv.
select_gate_traces(...) Stratified sample of silent connected-zero pairs plus
                       matched entangled controls, written as the
                       <stem>_gate_trace_selection.csv rppu_gate_trace.exe replays.
                       Also ``python -m data_analysis.select_gate_traces``.
dist_scaling_comparison(...) Overlay completed distance-scaling results for
                       circuit-ensemble or evolution-depth comparisons.
expvals(...)           Plot aggregate fermionic observables and Wick residuals.
tmi_collapse(...)      Fit and plot the TMI finite-size scaling collapse.
probe1_collapse(...)   Fit and plot the one-ancilla dynamical collapse.
probe2_collapse(...)   Fit and plot the two-ancilla bulk-exponent collapse, one
                       observable or, with ``stack_metrics=True``, all three
                       stacked flush on a shared x axis.
probe_distance_collapse(...) Plot and collapse mode-4 distance-resolved probes.
probe3_dynamics(...)   Plot three-probe mode-4 multipartite metrics and purity.
probe_anisotropy(...)  Plot mode-3 space/time correlators and estimate alpha.
free_energy_ceff(...)  Recreate Fig. 1(b) and extract the effective central charge.
free_energy_equilibration(...) Recreate Supplementary Fig. S1(a,b).
probe4_collapse(...)   Fit and plot four-ancilla I2/I3/I4 collapses.
probe_pc_collapse(...) Fit fixed-t p scans for one/two/four probes.
probe_pc_objective(...) Recreate the Zabalo Fig. S4 collapse-objective map.
probe_entropy_map(...) Plot the p-t S_Q heatmap and selected-p time curves.
front_velocity(...)    Fit one-probe mode-5 classical and entanglement front velocities.
stagger_velocity(...)  Fit two-probe mode-5 one-way velocities from insertion-delay
                       breakpoints.

All plotting functions return a dictionary containing the figure, axes, loaded
or summarized data, and fitted parameters. They show the plot by default; pass
``show=False`` to suppress ``plt.show()``.

Module layout
-------------
loading            file discovery, column lookup, run-metadata resolution
records            dist_scaling.exe's per-record binary: header, geometry table,
                   streaming reduction to curves and histograms, and the
                   summary CSV that caches it
summarize_records  the `./summarize_records` command line front end
plotting           figure helpers shared by every plot
fitting            generic weighted fits and error estimates
collapse           the shared one-parameter finite-size collapse engine
gmn, entropy, distance, expvals, tmi, free_energy
                   one module per observable family
probes_collapse    one-, two-, and four-probe bulk-exponent collapses
probes_dynamics    mode-4 distance-resolved and three-probe dynamics
probes_scans       mode-1 critical scans, mode-2 entropy maps, mode-3 anisotropy
front, stagger     mode-5 one-probe fronts and two-probe staggered velocities
pair_gap           dist_scaling.exe's connected-zero triple analysis and the
                   gate-path gap test
select_gate_traces the replay sample for rppu_gate_trace.exe
"""

from __future__ import annotations

from .distance import dist_percolation, dist_scaling, dist_scaling_comparison
from .pair_gap import dist_pair_gap, gate_path_gap_test, read_four_site, read_pair_summary
from .select_gate_traces import select_gate_traces
from .entropy import entropy
from .expvals import expvals
from .free_energy import free_energy_ceff, free_energy_equilibration
from .front import front_velocity
from .gmn import gmn
from .probes_collapse import probe1_collapse, probe2_collapse, probe4_collapse
from .probes_dynamics import probe3_dynamics, probe_distance_collapse
from .plotting import apply_paper_style
from .probes_scans import (
    probe_anisotropy,
    probe_entropy_map,
    probe_pc_collapse,
    probe_pc_objective,
)
from .stagger import stagger_velocity
from .tmi import tmi_collapse

__all__ = [
    "gmn",
    "entropy",
    "dist_percolation",
    "dist_pair_gap",
    "gate_path_gap_test",
    "select_gate_traces",
    "read_pair_summary",
    "read_four_site",
    "dist_scaling",
    "dist_scaling_comparison",
    "expvals",
    "tmi_collapse",
    "probe1_collapse",
    "probe2_collapse",
    "probe_distance_collapse",
    "probe3_dynamics",
    "probe_anisotropy",
    "free_energy_ceff",
    "free_energy_equilibration",
    "probe4_collapse",
    "probe_pc_collapse",
    "probe_pc_objective",
    "probe_entropy_map",
    "front_velocity",
    "stagger_velocity",
    "apply_paper_style",
]
