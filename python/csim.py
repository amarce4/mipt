import numpy as np
from parallel_mipt_rho3_writer_safe import write_mipt_rho3_bin_parallel_safe
from fermion import get_mipt_rho_1d_ferm

if __name__ == "__main__":
    n = 6
    d = 2 * n
    ps = np.linspace(0.0, 1.0, 21)
    realisations = 10000

    result = write_mipt_rho3_bin_parallel_safe(
        f"rho3_python_fermion_n_{n}.bin",
        get_mipt_rho_1d_ferm,
        n=n,
        d=d,
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