#pragma once

// Progress line for probe runs.
//
// A p scan is normally executed as one child process per p point, so the
// reporter accepts the global offsets and totals of the whole scan through
// MIPT_PROBED_GLOBAL_* and shows overall progress rather than the current
// child\'s slice.

#include "mipt/backend.hpp"
#include "mipt/env.hpp"
#include "mipt/util/text.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

namespace mipt::probed
{
using util::format_duration;


class ProgressReporter
{
  public:
    explicit ProgressReporter(std::uint64_t local_total)
        : global_offset_(static_cast<std::uint64_t>(
              env::integer("MIPT_PROBED_GLOBAL_OFFSET", 0, 0, std::numeric_limits<long>::max()))),
          global_total_(static_cast<std::uint64_t>(env::integer(
              "MIPT_PROBED_GLOBAL_TOTAL", static_cast<long>(local_total), 1, std::numeric_limits<long>::max()))),
          global_base_done_(static_cast<std::uint64_t>(
              env::integer("MIPT_PROBED_GLOBAL_BASE_DONE", 0, 0, std::numeric_limits<long>::max()))),
          global_start_ms_(static_cast<std::uint64_t>(
              env::integer("MIPT_PROBED_GLOBAL_START_MS", 0, 0, std::numeric_limits<long>::max()))),
          start_(std::chrono::steady_clock::now()), enabled_(env::boolean("MIPT_PROBED_PROGRESS", true)),
          interval_(env::integer("MIPT_PROBED_PROGRESS_MS", 500, 0, 60000))
    {
        last_ = start_ - interval_;
        point_start_ = start_;
    }

    void begin_point(std::uint64_t local_completed)
    {
        point_start_ = std::chrono::steady_clock::now();
        point_base_completed_ = local_completed;
    }

    void update(std::uint64_t local_completed, double p, std::size_t point_index, std::size_t point_count,
                bool force = false)
    {
        if (!enabled_)
        {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_ < interval_)
        {
            return;
        }
        last_ = now;
        printed_ = true;
        const double elapsed = std::chrono::duration<double>(now - start_).count();
        double average_rate = elapsed > 0.0 ? static_cast<double>(local_completed) / elapsed : 0.0;
        const double point_elapsed = std::chrono::duration<double>(now - point_start_).count();
        const std::uint64_t point_completed =
            local_completed >= point_base_completed_ ? local_completed - point_base_completed_ : 0;
        const double current_rate = point_elapsed > 0.0 ? static_cast<double>(point_completed) / point_elapsed : 0.0;
        const std::uint64_t global_completed = std::min(global_total_, global_offset_ + local_completed);
        const std::uint64_t remaining = global_total_ - global_completed;
        if (global_start_ms_ > 0)
        {
            const auto now_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                               std::chrono::system_clock::now().time_since_epoch())
                                                               .count());
            const double global_elapsed =
                now_ms > global_start_ms_ ? static_cast<double>(now_ms - global_start_ms_) / 1000.0 : 0.0;
            const std::uint64_t completed_this_run =
                global_completed > global_base_done_ ? global_completed - global_base_done_ : 0;
            if (global_elapsed > 0.0 && completed_this_run > 0)
            {
                average_rate = static_cast<double>(completed_this_run) / global_elapsed;
            }
        }
        const double eta = average_rate > 0.0 ? static_cast<double>(remaining) / average_rate
            : std::numeric_limits<double>::quiet_NaN();
        const std::size_t shown_index = static_cast<std::size_t>(env::integer(
            "MIPT_PROBED_GLOBAL_P_INDEX", static_cast<long>(point_index), 1, std::numeric_limits<long>::max()));
        const std::size_t shown_count = static_cast<std::size_t>(env::integer(
            "MIPT_PROBED_GLOBAL_P_COUNT", static_cast<long>(point_count), 1, std::numeric_limits<long>::max()));
        std::cout << "\rrealizations: " << global_completed << '/' << global_total_ << " | p=" << std::fixed
                  << std::setprecision(3) << p << " [" << shown_index << '/' << shown_count << ']'
                  << " | avg=" << std::setprecision(3) << average_rate << " traj/s"
                  << " | current=" << current_rate << " traj/s"
                  << " | ETA=" << format_duration(eta) << "     " << std::flush;
    }

    void finish() const
    {
        if (enabled_ && printed_)
        {
            std::cout << '\n';
        }
    }

  private:
    std::uint64_t global_offset_ = 0;
    std::uint64_t global_total_ = 0;
    std::uint64_t global_base_done_ = 0;
    std::uint64_t global_start_ms_ = 0;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_;
    std::chrono::steady_clock::time_point point_start_;
    std::uint64_t point_base_completed_ = 0;
    bool enabled_ = true;
    bool printed_ = false;
    std::chrono::milliseconds interval_{500};
};

} // namespace mipt::probed
