import numpy as np
from parallel_mipt_rho3_writer_safe import write_mipt_rho3_bin_parallel_safe

if __name__ == "__main__":
    n = 8
    d = 2 * n
    ps = np.linspace(0.8, 1.0, 5)
    realisations = 10000

    result = write_mipt_rho3_bin_parallel_safe(
        "rho3_python_archA2.bin",
        n=n,
        d=d,
        p_vals=ps,
        realisations=realisations,
        closed=True,

        # Start here:
        gpu_workers=1,
        cpu_workers=3,

        queue_size=64,
        stall_timeout_s=300,
        progress=True,
    )

    print(result)