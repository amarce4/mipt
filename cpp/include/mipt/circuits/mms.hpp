#pragma once

#include "mipt/types.hpp"

#include <cudaq.h>

#include <random>
#include <vector>

namespace mipt
{
struct MmsLayer
{
    int start = 0;

    std::vector<int> measure_flags;

    // Exactly one of these should be 1 for each qubit.
    std::vector<int> rot_x_flags;
    std::vector<int> rot_y_flags;
    std::vector<int> rot_xy_flags;
};


struct MmsKernel1D
{
    void operator()(int n,
                    const std::vector<MmsLayer> &layers,
                    bool closed) __qpu__
    {
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            int start = layers[layer].start;

            for (int i = start; i < n - 1; i += 2)
            {
                // Local rotation on q[i]
                if (layers[layer].rot_x_flags[i])
                {
                    rx(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_y_flags[i])
                {
                    ry(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_xy_flags[i])
                {
                    rz(-0.78539816339744830962, q[i]);
                    rx(1.57079632679489661923, q[i]);
                    rz(0.78539816339744830962, q[i]);
                }

                // Local rotation on q[i + 1]
                if (layers[layer].rot_x_flags[i + 1])
                {
                    rx(1.57079632679489661923, q[i + 1]);
                }

                if (layers[layer].rot_y_flags[i + 1])
                {
                    ry(1.57079632679489661923, q[i + 1]);
                }

                if (layers[layer].rot_xy_flags[i + 1])
                {
                    rz(-0.78539816339744830962, q[i + 1]);
                    rx(1.57079632679489661923, q[i + 1]);
                    rz(0.78539816339744830962, q[i + 1]);
                }

                // exp(-i*pi/4 X_i X_{i+1})
                h(q[i]);
                h(q[i + 1]);

                cx(q[i], q[i + 1]);
                rz(1.57079632679489661923, q[i + 1]);
                cx(q[i], q[i + 1]);

                h(q[i]);
                h(q[i + 1]);
            }

            // closed odd-layer closure: bond (n - 1, 0)
            if (closed && start == 1 && n > 2)
            {
                int i = n - 1;
                int j = 0;

                // Local rotation on q[n - 1]
                if (layers[layer].rot_x_flags[i])
                {
                    rx(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_y_flags[i])
                {
                    ry(1.57079632679489661923, q[i]);
                }

                if (layers[layer].rot_xy_flags[i])
                {
                    rz(-0.78539816339744830962, q[i]);
                    rx(1.57079632679489661923, q[i]);
                    rz(0.78539816339744830962, q[i]);
                }

                // Local rotation on q[0]
                if (layers[layer].rot_x_flags[j])
                {
                    rx(1.57079632679489661923, q[j]);
                }

                if (layers[layer].rot_y_flags[j])
                {
                    ry(1.57079632679489661923, q[j]);
                }

                if (layers[layer].rot_xy_flags[j])
                {
                    rz(-0.78539816339744830962, q[j]);
                    rx(1.57079632679489661923, q[j]);
                    rz(0.78539816339744830962, q[j]);
                }

                // exp(-i*pi/4 X_{n-1} X_0)
                h(q[i]);
                h(q[j]);

                cx(q[i], q[j]);
                rz(1.57079632679489661923, q[j]);
                cx(q[i], q[j]);

                h(q[i]);
                h(q[j]);
            }

            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[i])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

struct MmsKernel2D
{
    void operator()(int x, int y,
                    const std::vector<MmsLayer> &layers,
                    bool closed) __qpu__
    {
        int n = x * y;
        cudaq::qvector q(n);

        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            int start = layers[layer].start;

            if (layer % 4 == 0 || layer % 4 == 1)
            {
                for (int i = 0; i < y; ++i)
                {
                    for (int j = start; j < x-1; j += 2)
                    {
                        int idx = i * x + j;
                        // Local rotation on q[idx]
                        if (layers[layer].rot_x_flags[idx])
                        {
                            rx(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_y_flags[idx])
                        {
                            ry(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_xy_flags[idx])
                        {
                            rz(-0.78539816339744830962, q[idx]);
                            rx(1.57079632679489661923, q[idx]);
                            rz(0.78539816339744830962, q[idx]);
                        }

                        // Local rotation on q[idx + 1]
                        if (layers[layer].rot_x_flags[idx + 1])
                        {
                            rx(1.57079632679489661923, q[idx + 1]);
                        }

                        if (layers[layer].rot_y_flags[idx + 1])
                        {
                            ry(1.57079632679489661923, q[idx + 1]);
                        }

                        if (layers[layer].rot_xy_flags[idx + 1])
                        {
                            rz(-0.78539816339744830962, q[idx + 1]);
                            rx(1.57079632679489661923, q[idx + 1]);
                            rz(0.78539816339744830962, q[idx + 1]);
                        }

                        // exp(-i*pi/4 X_i X_{i+1})
                        h(q[idx]);
                        h(q[idx + 1]);

                        cx(q[idx], q[idx + 1]);
                        rz(1.57079632679489661923, q[idx + 1]);
                        cx(q[idx], q[idx + 1]);

                        h(q[idx]);
                        h(q[idx + 1]);
                    }

                    // closed odd-layer closure: bond (n - 1, 0)
                    if (closed && start == 1 && x > 2)
                    {
                        int k = i*x;
                        int l = i*x + x - 1;

                        // Local rotation on q[k]
                        if (layers[layer].rot_x_flags[k])
                        {
                            rx(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_y_flags[k])
                        {
                            ry(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_xy_flags[k])
                        {
                            rz(-0.78539816339744830962, q[k]);
                            rx(1.57079632679489661923, q[k]);
                            rz(0.78539816339744830962, q[k]);
                        }

                        // Local rotation on q[0]
                        if (layers[layer].rot_x_flags[l])
                        {
                            rx(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_y_flags[l])
                        {
                            ry(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_xy_flags[l])
                        {
                            rz(-0.78539816339744830962, q[l]);
                            rx(1.57079632679489661923, q[l]);
                            rz(0.78539816339744830962, q[l]);
                        }

                        // exp(-i*pi/4 X_{n-1} X_0)
                        h(q[k]);
                        h(q[l]);

                        cx(q[k], q[l]);
                        rz(1.57079632679489661923, q[l]);
                        cx(q[k], q[l]);

                        h(q[k]);
                        h(q[l]);
                    }
                }
            }
            else
            {
                for (int i = 0; i < x; ++i)
                {
                    int x_start = (i % 2 == 0) ? 0 : 1;
                    x_start = (x_start + layers[layer].start) % 2;
                    for (int j = x_start; j < y-1; j += 2)
                    {
                        int idx = j * y + i;
                        int idx_next = (j + 1) * y + i;
                        // Local rotation on q[idx]
                        if (layers[layer].rot_x_flags[idx])
                        {
                            rx(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_y_flags[idx])
                        {
                            ry(1.57079632679489661923, q[idx]);
                        }

                        if (layers[layer].rot_xy_flags[idx])
                        {
                            rz(-0.78539816339744830962, q[idx]);
                            rx(1.57079632679489661923, q[idx]);
                            rz(0.78539816339744830962, q[idx]);
                        }

                        // Local rotation on q[idx_next]
                        if (layers[layer].rot_x_flags[idx_next])
                        {
                            rx(1.57079632679489661923, q[idx_next]);
                        }

                        if (layers[layer].rot_y_flags[idx_next])
                        {
                            ry(1.57079632679489661923, q[idx_next]);
                        }

                        if (layers[layer].rot_xy_flags[idx_next])
                        {
                            rz(-0.78539816339744830962, q[idx_next]);
                            rx(1.57079632679489661923, q[idx_next]);
                            rz(0.78539816339744830962, q[idx_next]);
                        }

                        // exp(-i*pi/4 X_i X_{i+1})
                        h(q[idx]);
                        h(q[idx_next]);

                        cx(q[idx], q[idx_next]);
                        rz(1.57079632679489661923, q[idx_next]);
                        cx(q[idx], q[idx_next]);

                        h(q[idx]);
                        h(q[idx_next]);
                    }

                    // closed odd-layer closure: bond (n - 1, 0)
                    if (closed && x_start == 1 && y > 2)
                    {
                        int k = i;
                        int l = (y-1)*y + i;

                        // Local rotation on q[k]
                        if (layers[layer].rot_x_flags[k])
                        {
                            rx(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_y_flags[k])
                        {
                            ry(1.57079632679489661923, q[k]);
                        }

                        if (layers[layer].rot_xy_flags[k])
                        {
                            rz(-0.78539816339744830962, q[k]);
                            rx(1.57079632679489661923, q[k]);
                            rz(0.78539816339744830962, q[k]);
                        }

                        // Local rotation on q[0]
                        if (layers[layer].rot_x_flags[l])
                        {
                            rx(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_y_flags[l])
                        {
                            ry(1.57079632679489661923, q[l]);
                        }

                        if (layers[layer].rot_xy_flags[l])
                        {
                            rz(-0.78539816339744830962, q[l]);
                            rx(1.57079632679489661923, q[l]);
                            rz(0.78539816339744830962, q[l]);
                        }

                        // exp(-i*pi/4 X_{n-1} X_0)
                        h(q[k]);
                        h(q[l]);

                        cx(q[k], q[l]);
                        rz(1.57079632679489661923, q[l]);
                        cx(q[k], q[l]);

                        h(q[k]);
                        h(q[l]);
                    }
                }
            }
            for (int i = 0; i < n; ++i)
            {
                if (layers[layer].measure_flags[i])
                {
                    mz(q[i]);
                }
            }
        }
    }
};

inline void fill_mms_layer(MmsLayer &layer,
                     int n,
                     int start,
                     double p,
                     std::mt19937 &rng)
{
    std::uniform_int_distribution<int> axis_dist(0, 2);
    std::bernoulli_distribution measure_dist(p);

    layer.start = start;
    layer.measure_flags.resize(n);
    layer.rot_x_flags.resize(n);
    layer.rot_y_flags.resize(n);
    layer.rot_xy_flags.resize(n);

    for (int q = 0; q < n; ++q)
    {
        const int axis = axis_dist(rng);
        layer.rot_x_flags[q] = (axis == 0) ? 1 : 0;
        layer.rot_y_flags[q] = (axis == 1) ? 1 : 0;
        layer.rot_xy_flags[q] = (axis == 2) ? 1 : 0;

        if (p <= 0.0)
        {
            layer.measure_flags[q] = 0;
        }
        else if (p >= 1.0)
        {
            layer.measure_flags[q] = 1;
        }
        else
        {
            layer.measure_flags[q] = measure_dist(rng) ? 1 : 0;
        }
    }
}

inline MmsLayer make_mms_layer(int n,
                          int start,
                          double p,
                          std::mt19937 &rng)
{
    MmsLayer layer;
    fill_mms_layer(layer, n, start, p, rng);
    return layer;
}

inline void build_mms_layers(std::vector<MmsLayer> &layers,
                           int n,
                           int periods,
                           double p,
                           std::mt19937 &rng)
{
    std::bernoulli_distribution extra_even_layer(0.5);
    const bool add_extra_even_layer = extra_even_layer(rng);
    const std::size_t layer_count = static_cast<std::size_t>(2 * periods + (add_extra_even_layer ? 1 : 0));
    layers.resize(layer_count);

    std::size_t idx = 0;
    for (int period = 0; period < periods; ++period)
    {
        fill_mms_layer(layers[idx++], n, 0, p, rng);
        fill_mms_layer(layers[idx++], n, 1, p, rng);
    }
    if (add_extra_even_layer)
    {
        fill_mms_layer(layers[idx++], n, 0, p, rng);
    }
}

inline std::vector<MmsLayer> make_mms_layers(int n,
                                     int periods,
                                     double p)
{
    std::mt19937 rng(std::random_device{}());
    std::vector<MmsLayer> layers;
    layers.reserve(2 * periods + 1);
    build_mms_layers(layers, n, periods, p, rng);
    return layers;
}

inline CircuitWorkStats circuit_work_stats_mms(const std::vector<MmsLayer> &layers, int n, bool closed)
{
    CircuitWorkStats stats;
    stats.layers = layers.size();
    for (const auto &layer : layers)
    {
        for (int flag : layer.measure_flags)
        {
            stats.measurements += (flag != 0) ? 1u : 0u;
        }
        const int start = layer.start;
        for (int i = start; i < n - 1; i += 2)
        {
            ++stats.mms_bonds;
        }
        if (closed && start == 1 && n > 2)
        {
            ++stats.mms_bonds;
        }
    }
    stats.logical_two_site_gates = stats.mms_bonds;
    // Each MMS bond is implemented as two CX gates plus single-qubit basis rotations.
    stats.estimated_cudaq_two_qubit_ops = 2u * stats.mms_bonds;
    return stats;
}

} // namespace mipt
