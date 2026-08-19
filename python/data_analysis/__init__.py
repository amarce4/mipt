"""Reusable analysis and plotting tools for the MIPT simulation outputs.

Public functions
----------------
gmn(...)               Plot GMN/fGMN and optional minimum bipartite negativity.
entropy(...)           Plot entropy scaling curves and weighted fits; ``order=1``
                       selects von Neumann, ``order=2`` (default) Rényi-2.
dist_scaling(...)      Plot two-site (MI/negativity) or three-site (TMI/GMN)
                       distance scaling and power-law fits; ``ent_decomp=True``
                       splits the pair negativities into the entangled fraction
                       and the conditional magnitude instead.
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
plotting           figure helpers shared by every plot
fitting            generic weighted fits and error estimates
collapse           the shared one-parameter finite-size collapse engine
gmn, entropy, distance, expvals, tmi, free_energy
                   one module per observable family
probes_collapse    one-, two-, and four-probe bulk-exponent collapses
probes_dynamics    mode-4 distance-resolved and three-probe dynamics
probes_scans       mode-1 critical scans, mode-2 entropy maps, mode-3 anisotropy
front, stagger     mode-5 one-probe fronts and two-probe staggered velocities
"""

from __future__ import annotations

from .distance import dist_scaling
from .entropy import entropy
from .expvals import expvals
from .free_energy import free_energy_ceff, free_energy_equilibration
from .front import front_velocity
from .gmn import gmn
from .probes_collapse import probe1_collapse, probe2_collapse, probe4_collapse
from .probes_dynamics import probe3_dynamics, probe_distance_collapse
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
    "dist_scaling",
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
]
