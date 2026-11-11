// Offline graph replay for dist_scaling.exe k=2 record files.
//
//   ./dist_graph_replay.exe augment <in_records.bin> [out_records.bin]
//   ./dist_graph_replay.exe paths <records.bin> <selection.csv> [out.csv]
//
// Regenerates every trajectory's spacetime graph from the master seed in the
// record header -- host only: no GPU, no CUDA-Q, no state vector -- checks it
// against the stored records, and either appends the temporal-turn and
// doubled-network observables (`augment`, producing what dist_scaling.exe
// writes with MIPT_DIST_TURN_DIAGNOSTICS=1) or spells out the canonical paths
// of selected pairs (`paths`). include/mipt/graph_replay.hpp holds the logic,
// the reasoning, and the refusals, so `make test-dist` pins all of it; this
// file is only the entry point.
//
// Exit status: 0 success; 1 I/O or other error; 2 usage error or refused
// input; 3 validation failure (nothing is left at the output path).

#include "mipt/graph_replay.hpp"

int main(int argc, char **argv)
{
    return mipt::dist::graph_replay::cli_main(argc, argv);
}
