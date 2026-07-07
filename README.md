# Tripartite Entanglement in MIPT Architectures

This repository currently studies the following:
## Bosonic (Qubit) MIPT
- MMS (C++) and Random Haar Unitary (Python) Entangling Gates
    - `mipt.py`, `csim.py`, `mipt.cpp`
- Genuine Multipartite Negativity (GMN) via SDP
    - `gmn.py`, `gmn.hpp`, `gmn_main.cpp`, `gmn_mosek_c.c`
- Genuine Network Multipartite Entanglement (GNME) via Ring Inflation SDP
    - `gnme.py`, `ring_inflation.cpp`
- 2D Brickwork Entangling Systems
    - `mipt.py`, `mipt.cpp`

## Fermionic (Majorana) MIPT
- Random Parity-Preserving Unitary (RPPU) Entangling Gates
    - `fermion.py`, `csim.py`, `mipt.cpp`
- Reduced Fermionic Gate Set (RFGS)
    - `mipt_fermion.cpp`
- TMI Determination of Critical Point
    - `tmi.cpp`, `sim_tmi.cpp`

## Installation
- Python (venv)
```shell
pip install requirements.txt
```
- C++
```shell
chmod +x cpp_install && ./cpp_install
```