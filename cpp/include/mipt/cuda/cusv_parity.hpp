#pragma once

// Device passes for the parity-sector encoding of the cuStateVec engine.
//
// A parity-preserving circuit started from |0...0> never leaves the even
// global-parity sector: every basis state in the support has even Hamming
// weight, so the odd-parity half of the state vector is identically zero for
// the whole trajectory.  The encoded representation drops it.  An encoded
// index `e` carries bits b_0..b_{n-2} and the missing bit is recovered as
//
//     b_{n-1} = parity(e),
//
// so the state is 2^(n-1) amplitudes rather than 2^n and every pass moves half
// the bytes.  See mipt/cusv/parity.hpp for the operator-level story.
//
// Gates that touch only qubits 0..n-2 need nothing from this file: a
// parity-preserving gate acts on the encoded index exactly as it acts on the
// full one, so custatevecApplyMatrix drives them unchanged.  The four passes
// below are the ones that cannot be expressed that way.

#include <cstdint>

extern "C" {

// A parity-preserving two-mode gate on (target, n-1) becomes a 2x2 on the
// encoded bit `target` whose matrix is selected by
//     s = parity(e with bit `target` cleared),
// because b_{n-1} = b_target XOR s.  `even_ri` is the 2x2 used at s = 0 and
// `odd_ri` the one used at s = 1, each 8 doubles: row-major (re, im) pairs
// ordered M00, M01, M10, M11, with row/column indexed by the value of
// b_target.
int mipt_cuda_parity_gate_f32(void *device_state_vector, int encoded_qubits, int target,
                              const double *even_ri, const double *odd_ri);
int mipt_cuda_parity_gate_f64(void *device_state_vector, int encoded_qubits, int target,
                              const double *even_ri, const double *odd_ri);

// Measuring site n-1 means measuring parity(e).  Writes the two bin weights
// {even, odd} to `host_weights`.  Block partials are summed in a fixed order
// rather than with atomics, so the result does not depend on scheduling.
int mipt_cuda_parity_abs2_f32(const void *device_state_vector, int encoded_qubits,
                              double *host_weights);
int mipt_cuda_parity_abs2_f64(const void *device_state_vector, int encoded_qubits,
                              double *host_weights);

// Projects onto parity(e) == outcome and rescales by 1/sqrt(weight).
int mipt_cuda_parity_collapse_f32(void *device_state_vector, int encoded_qubits, int outcome,
                                  double weight);
int mipt_cuda_parity_collapse_f64(void *device_state_vector, int encoded_qubits, int outcome,
                                  double weight);

// Expands the encoded state sitting in the low half of a 2^n buffer into the
// full 2^n state vector, in place.  Amplitude `e` moves to `e | 2^(n-1)` when
// parity(e) is odd and stays put otherwise, so every thread owns one source
// and two destinations and no two threads collide.
int mipt_cuda_parity_expand_f32(void *device_state_vector, int full_qubits);
int mipt_cuda_parity_expand_f64(void *device_state_vector, int full_qubits);

const char *mipt_cuda_cusv_parity_last_error();

} // extern "C"
