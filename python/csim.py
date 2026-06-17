import numpy as np
from parallel_mipt_rho3_writer_safe import write_mipt_rho3_bin_parallel_safe
from fermion import get_mipt_rho_1d_ferm
from fermionic_partial_trace import make_ring_3mode_plans

if __name__ == "__main__":
    n = 8
    d = 2 * n
    ps = np.linspace(0.0, 1.0, 21)
    realisations = 5000

    plans = make_ring_3mode_plans(n, closed=True, basis_order="little")

    result = write_mipt_rho3_bin_parallel_safe(
        f"bin/rho3_python_fermion_n_{n}_real_{realisations}.bin",
        get_mipt_rho_1d_ferm,
        n=n,
        d=d,
        plans=plans,
        p_vals=ps,
        realisations=realisations,
        closed=True,

        # Start here:
        gpu_workers=0,
        cpu_workers=4,

        queue_size=64,
        stall_timeout_s=300,
        progress=True,
    )

    print(result)