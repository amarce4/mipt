#pragma once

// Per-p subprocess isolation, checkpointing, and resume.
//
// A long p scan runs each point in a fresh child process. That keeps CUDA-Q
// and cuStateVec runtime state from accumulating across a scan, and it means a
// crash costs one p point rather than the whole run: every completed point is
// appended to the output CSV atomically, and rerunning the identical command
// skips the points already present.

#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/types.hpp"
#include "mipt/util/text.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt::probed
{
using util::precise_decimal;
using util::shell_quote;

inline bool same_probability(double lhs, double rhs)
{
    return std::abs(lhs - rhs) <= 1.0e-12 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

// Split one CSV line, honouring the double quotes around text fields such as
// circuit_name.
inline std::vector<std::string> split_csv_line(const std::string &line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i)
    {
        const char c = line[i];
        if (quoted)
        {
            if (c == '"')
            {
                if (i + 1 < line.size() && line[i + 1] == '"')
                {
                    field.push_back('"');
                    ++i;
                }
                else
                {
                    quoted = false;
                }
            }
            else
            {
                field.push_back(c);
            }
        }
        else if (c == '"')
        {
            quoted = true;
        }
        else if (c == ',')
        {
            fields.push_back(field);
            field.clear();
        }
        else
        {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

// The p values already present in a checkpoint. The column is located by name
// rather than by position, so the self-describing metadata prefix can precede
// it without breaking resume of files written by earlier builds.
inline std::vector<double> completed_probabilities(const std::string &path)
{
    std::ifstream csv(path);
    if (!csv)
    {
        return {};
    }

    std::string line;
    if (!std::getline(csv, line))
    {
        return {};
    }
    const auto header = split_csv_line(line);
    const auto column = std::find(header.begin(), header.end(), "p");
    if (column == header.end())
    {
        throw std::runtime_error("Cannot resume from a checkpoint CSV without a p column: " + path);
    }
    const auto p_index = static_cast<std::size_t>(std::distance(header.begin(), column));

    std::vector<double> completed;
    while (std::getline(csv, line))
    {
        if (line.empty())
        {
            continue;
        }
        const auto fields = split_csv_line(line);
        if (fields.size() <= p_index)
        {
            throw std::runtime_error("Cannot resume from malformed checkpoint row in: " + path);
        }
        const double p = std::stod(fields[p_index]);
        if (std::none_of(completed.begin(), completed.end(), [p](double prior) { return same_probability(p, prior); }))
        {
            completed.push_back(p);
        }
    }
    return completed;
}

// Append one finished point to the checkpoint through a temporary file and a
// rename, so an interrupted append can never truncate completed work.
inline void append_checkpoint_csv(const std::string &point_path, const std::string &output_path)
{
    std::ifstream point(point_path);
    if (!point)
    {
        throw std::runtime_error("Isolated point did not produce its CSV: " + point_path);
    }

    std::string point_header;
    if (!std::getline(point, point_header) || point_header.empty())
    {
        throw std::runtime_error("Isolated point produced an empty CSV: " + point_path);
    }

    std::ifstream existing(output_path);
    std::string existing_header;
    const bool append = existing && static_cast<bool>(std::getline(existing, existing_header));
    if (append && existing_header != point_header)
    {
        throw std::runtime_error(
            "Checkpoint header does not match the new output: " + output_path +
            "\n  existing: " + existing_header + "\n  new:      " + point_header +
            "\nA checkpoint written before the self-describing metadata columns "
            "(N,circ_type,circuit_name,realizations) were added cannot be extended by this build. "
            "Finish the scan with the older binary, or move the old file aside and rerun to start fresh.");
    }

    const std::string update_path = output_path + ".update.tmp";
    std::ofstream output(update_path, std::ios::out | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("Could not update checkpoint CSV: " + output_path);
    }
    if (append)
    {
        std::ifstream prior(output_path);
        output << prior.rdbuf();
        if (!prior.eof() && prior.fail())
        {
            throw std::runtime_error("Could not read checkpoint CSV: " + output_path);
        }
    }
    else
    {
        output << point_header << '\n';
    }
    output << point.rdbuf();
    output.flush();
    if (!output)
    {
        throw std::runtime_error("Failed while updating checkpoint CSV: " + output_path);
    }
    output.close();
    if (std::rename(update_path.c_str(), output_path.c_str()) != 0)
    {
        throw std::runtime_error("Could not atomically replace checkpoint CSV: " + output_path);
    }
}

inline int run_isolated_p_scan(const char *program, int probes, int n, int realizations, CircuitType type, int mode,
                               const std::vector<double> &p_values, const std::vector<int> &times,
                               const std::string &output)
{
    const bool resume = env::boolean("MIPT_PROBED_RESUME", true);
    if (!resume)
    {
        std::remove(output.c_str());
    }
    auto completed = completed_probabilities(output);
    const int retries = static_cast<int>(env::integer("MIPT_PROBED_RETRIES", 1, 0, 10));

    const std::uint64_t global_total =
        static_cast<std::uint64_t>(p_values.size()) * static_cast<std::uint64_t>(realizations);
    const std::uint64_t global_base_done =
        static_cast<std::uint64_t>(completed.size()) * static_cast<std::uint64_t>(realizations);
    const auto global_start_ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
    if (!completed.empty())
    {
        const std::uint64_t done =
            static_cast<std::uint64_t>(completed.size()) * static_cast<std::uint64_t>(realizations);
        std::cout << "Resuming: " << completed.size() << '/' << p_values.size()
                  << " p points complete; realizations done=" << done
                  << ", left=" << global_total - std::min(global_total, done) << ".\n";
    }
    for (std::size_t index = 0; index < p_values.size(); ++index)
    {
        const double p = p_values[index];
        const bool already_done =
            std::any_of(completed.begin(), completed.end(), [p](double prior) { return same_probability(p, prior); });
        if (already_done)
        {
            backend::verbose_log("p point " + std::to_string(index + 1) + '/' + std::to_string(p_values.size()) +
                                 " already checkpointed");
            continue;
        }

        const std::string point_output = output + ".point_" + std::to_string(index) + ".tmp";
        std::ostringstream base;
        const std::uint64_t global_offset =
            static_cast<std::uint64_t>(completed.size()) * static_cast<std::uint64_t>(realizations);
        base << "MIPT_PROBED_INTERNAL_CHILD=1 "
             << "MIPT_PROBED_ISOLATE_P=0 "
             << "MIPT_PROBED_GLOBAL_OFFSET=" << global_offset << ' ' << "MIPT_PROBED_GLOBAL_TOTAL=" << global_total
             << ' ' << "MIPT_PROBED_GLOBAL_BASE_DONE=" << global_base_done << ' '
             << "MIPT_PROBED_GLOBAL_START_MS=" << global_start_ms << ' ' << "MIPT_PROBED_GLOBAL_P_INDEX=" << index + 1
             << ' ' << "MIPT_PROBED_GLOBAL_P_COUNT=" << p_values.size() << ' '
             << "MIPT_PROBED_INTERNAL_OUTPUT=" << shell_quote(point_output) << ' ' << shell_quote(program) << ' '
             << probes << ' ' << n << ' ' << realizations << ' ' << static_cast<int>(type) << ' ' << mode;
        if (mode == 1)
        {
            base << ' ' << precise_decimal(p) << ' ' << precise_decimal(p) << " 1";
        }
        else
        {
            base << ' ' << times.front() << ' ' << times.back() << ' ' << times.size() << ' ' << precise_decimal(p)
                 << ' ' << precise_decimal(p) << " 1";
        }

        backend::verbose_log("starting p point " + std::to_string(index + 1) + '/' + std::to_string(p_values.size()) +
                             " p=" + precise_decimal(p));
        int status = -1;
        for (int attempt = 0; attempt <= retries; ++attempt)
        {
            std::remove(point_output.c_str());
            std::string command;
            if (attempt > 0)
            {
                command = "MIPT_PROBED_PREFETCH=0 ";
                std::cerr << "Retrying p=" << precise_decimal(p) << " in a fresh process with CPU prefetch disabled"
                          << " (attempt " << attempt + 1 << '/' << retries + 1 << ").\n";
            }
            command += base.str();
            status = std::system(command.c_str());
            if (status == 0)
            {
                break;
            }
        }
        if (status != 0)
        {
            throw std::runtime_error("Isolated simulation failed for p=" + precise_decimal(p) + " after " +
                                     std::to_string(retries + 1) +
                                     " attempt(s). Earlier p points remain checkpointed; rerun the "
                                     "same command to resume.");
        }

        append_checkpoint_csv(point_output, output);
        std::remove(point_output.c_str());
        completed.push_back(p);
        backend::verbose_log("checkpointed p=" + precise_decimal(p) + " to " + output);
    }

    std::cout << "Saved " << p_values.size() * times.size() << " rows to " << output << ".\n";
    return 0;
}

} // namespace mipt::probed
