#pragma once

// Deferred GMN/fGMN evaluation for three-probe runs.
//
// The SDP solve is far slower than the circuit trajectory that produced its
// density matrix, so RDMs are batched and solved on background threads while
// the simulation continues. Results are folded back into the aggregate they
// came from, which is why the queue holds a reference to the aggregate vector.

#include "mipt/analysis/fgmn.hpp"
#include "mipt/probed/sample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <future>
#include <limits>
#include <utility>
#include <vector>

namespace mipt::probed
{

struct SdpJob
{
    std::size_t aggregate_index = 0;
    std::array<double, 128> rho{};
};

struct SdpResult
{
    std::size_t aggregate_index = 0;
    double value = std::numeric_limits<double>::quiet_NaN();
};

class SdpBatchQueue
{
  public:
    SdpBatchQueue(std::vector<Aggregate> &aggregates, bool fermionic, std::size_t batch_size,
                  std::size_t max_pending_batches)
        : aggregates_(aggregates), fermionic_(fermionic), batch_size_(std::max<std::size_t>(1, batch_size)),
          max_pending_batches_(std::max<std::size_t>(1, max_pending_batches))
    {
        batch_.reserve(batch_size_);
    }

    void submit(SdpJob job)
    {
        batch_.push_back(std::move(job));
        if (batch_.size() >= batch_size_)
        {
            dispatch();
        }
    }

    void finish()
    {
        if (!batch_.empty())
        {
            dispatch();
        }
        while (!pending_.empty())
        {
            collect_front();
        }
    }

  private:
    void dispatch()
    {
        while (pending_.size() >= max_pending_batches_)
        {
            collect_front();
        }
        std::vector<SdpJob> jobs;
        jobs.swap(batch_);
        batch_.reserve(batch_size_);
        const bool fermionic = fermionic_;
        pending_.push_back(std::async(std::launch::async, [jobs = std::move(jobs), fermionic]() mutable {
            std::vector<SdpResult> results;
            results.reserve(jobs.size());
            for (auto &job : jobs)
            {
                const double value = fermionic ? compute_fgmn_mosek_8x8_cpp(job.rho.data())
                                               : compute_gmn_mosek_complex_8x8_cpp(job.rho.data());
                results.push_back({job.aggregate_index, value});
            }
            return results;
        }));
    }

    void collect_front()
    {
        auto results = pending_.front().get();
        pending_.pop_front();
        for (const auto &result : results)
        {
            auto &aggregate = aggregates_.at(result.aggregate_index);
            if (std::isfinite(result.value))
            {
                aggregate.gmn.add(std::max(0.0, result.value));
            }
            else
            {
                ++aggregate.gmn_solver_failures;
            }
        }
    }

    std::vector<Aggregate> &aggregates_;
    bool fermionic_ = false;
    std::size_t batch_size_ = 1;
    std::size_t max_pending_batches_ = 1;
    std::vector<SdpJob> batch_;
    std::deque<std::future<std::vector<SdpResult>>> pending_;
};

} // namespace mipt::probed
