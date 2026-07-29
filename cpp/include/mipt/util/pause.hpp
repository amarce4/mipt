#pragma once

// Cooperative host pause.
//
// Creating the sentinel file — PAUSE_MIPT in the working directory by default —
// makes a running simulation stop at its next safe checkpoint: after the
// current trajectory or batch, with output already flushed. Removing the file
// resumes it. This is how a long run yields the GPU without losing its
// completed work.
//
// Paused time is accumulated separately so progress rates and ETAs can exclude
// it; callers that keep their own "last report" timestamp should advance it by
// the returned duration.
//
// Every executable prints exactly:
//
//     PAUSED by host sentinel PAUSE_MIPT; remove it to resume.
//     RESUMED after 37m 28s.

#include "mipt/env.hpp"
#include "mipt/util/text.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

namespace mipt::util
{

class PauseSentinel
{
  public:
    // `legacy_env` keeps an executable's pre-existing override working;
    // MIPT_PAUSE_FILE applies to every executable and is checked second.
    // Setting either to 0, none, or disabled turns the sentinel off.
    explicit PauseSentinel(const char *legacy_env = nullptr) : path_(resolve_path(legacy_env)) {}

    bool enabled() const
    {
        return !path_.empty();
    }

    const std::string &path() const
    {
        return path_;
    }

    double paused_seconds() const
    {
        return paused_seconds_;
    }

    bool paused() const
    {
        return enabled() && exists(path_);
    }

    // Block while the sentinel exists. `flush` runs once before the wait so
    // partial results reach disk. Returns the length of this pause in seconds,
    // or 0 if there was none.
    template <typename Flush>
    double wait(Flush flush)
    {
        if (!paused())
        {
            return 0.0;
        }

        flush();
        const auto start = std::chrono::steady_clock::now();
        std::cerr << "\nPAUSED by host sentinel " << path_ << "; remove it to resume.\n";
        while (exists(path_))
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        const double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        paused_seconds_ += duration;
        std::cerr << "RESUMED after " << format_duration(duration, DurationStyle::Compact) << ".\n";
        return duration;
    }

    double wait()
    {
        return wait([] {});
    }

    // One line for a startup banner.
    std::string describe() const
    {
        return enabled() ? "host_pause_file=" + path_ + " (create it to pause, remove it to resume)"
                         : std::string("host_pause_control=disabled");
    }

  private:
    static bool exists(const std::string &path)
    {
        std::error_code error;
        return !path.empty() && std::filesystem::exists(path, error) && !error;
    }

    static std::string resolve_path(const char *legacy_env)
    {
        for (const char *name : {legacy_env, "MIPT_PAUSE_FILE"})
        {
            if (name == nullptr)
            {
                continue;
            }
            const char *value = std::getenv(name);
            if (env::disables_path(value))
            {
                return {};
            }
            if (value != nullptr && *value != '\0')
            {
                return std::string(value);
            }
        }
        return "PAUSE_MIPT";
    }

    std::string path_;
    double paused_seconds_ = 0.0;
};

} // namespace mipt::util
