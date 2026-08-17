#pragma once

// Fatal-signal reporting for the executables that drive MOSEK Fusion.
//
// Fusion crashes *natively*: it corrupts memory and dies on a signal rather
// than throwing, so no try/catch sees it and the process simply vanishes. On a
// run that has been going for hours that leaves nothing to act on -- not the
// concurrency it was using, not which file to resume, not even which solver was
// running. This prints that, once, from the signal handler.
//
// **Everything here has to be async-signal-safe.** The handler can interrupt
// any instruction, including one inside malloc, so it uses only ::write, text
// captured before the run, and lock-free atomics. No stdio, no std::string, no
// formatting, no allocation. That is why the context is snapshotted into a
// fixed buffer by `set_context` instead of being built when it is needed.
//
// Counters are watched by pointer rather than copied, so the numbers printed
// are the live ones at the instant of the crash. They must be lock-free
// atomics: the handler usually runs on the thread that died, which is not the
// thread that owns them.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <csignal>
#include <cstdlib>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace mipt::util::crash
{

inline constexpr std::size_t CONTEXT_CAPACITY = 4096;
inline constexpr std::size_t MAX_COUNTERS = 8;

static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
              "Watched counters are read from a signal handler, so they must be lock-free.");

struct WatchedCounter
{
    const char *label = nullptr;
    const std::atomic<std::uint64_t> *value = nullptr;
};

inline char g_context[CONTEXT_CAPACITY] = {};
inline std::atomic<std::size_t> g_context_length{0};
inline WatchedCounter g_counters[MAX_COUNTERS] = {};
inline std::atomic<std::size_t> g_counter_count{0};
inline const char *g_program = "mipt";

namespace detail
{

inline void write_all(const char *data, std::size_t length)
{
#if !defined(_WIN32)
    while (length > 0)
    {
        const ssize_t written = ::write(STDERR_FILENO, data, length);
        if (written <= 0)
        {
            return;
        }
        data += written;
        length -= static_cast<std::size_t>(written);
    }
#else
    (void)data;
    (void)length;
#endif
}

// std::strlen is not on the POSIX async-signal-safe list, so count by hand.
inline void write_text(const char *text)
{
    if (text == nullptr)
    {
        return;
    }
    std::size_t length = 0;
    while (text[length] != '\0')
    {
        ++length;
    }
    write_all(text, length);
}

inline void write_uint(std::uint64_t value)
{
    char buffer[24];
    std::size_t index = sizeof(buffer);
    do
    {
        buffer[--index] = static_cast<char>('0' + static_cast<char>(value % 10u));
        value /= 10u;
    } while (value != 0);
    write_all(buffer + index, sizeof(buffer) - index);
}

inline const char *signal_name(int number)
{
    switch (number)
    {
    case SIGSEGV:
        return "SIGSEGV (invalid memory access)";
    case SIGABRT:
        return "SIGABRT (abort)";
    case SIGILL:
        return "SIGILL (illegal instruction)";
    case SIGFPE:
        return "SIGFPE (arithmetic fault)";
#if defined(SIGBUS)
    case SIGBUS:
        return "SIGBUS (bad address)";
#endif
    default:
        return "fatal signal";
    }
}

inline void handler(int number)
{
    write_text("\n\n=== ");
    write_text(g_program);
    write_text(": fatal native crash, ");
    write_text(signal_name(number));
    write_text(" ===\n");

    const std::size_t length = g_context_length.load(std::memory_order_relaxed);
    if (length > 0)
    {
        write_all(g_context, length);
        write_text("\n");
    }

    const std::size_t counters = g_counter_count.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < counters; ++i)
    {
        if (g_counters[i].value == nullptr)
        {
            continue;
        }
        write_text("  ");
        write_text(g_counters[i].label);
        write_text("=");
        write_uint(g_counters[i].value->load(std::memory_order_relaxed));
        write_text("\n");
    }

    write_text(
        "\nThis is a native crash inside the solver, not a C++ exception, so nothing\n"
        "could catch it. The usual cause is MOSEK Fusion under concurrent solves:\n"
        "lower FGMN_MAX_CONCURRENT_MOSEK (and MIPT_DIST_GMN_PENDING_BATCHES /\n"
        "MIPT_PROBED_GMN_PENDING_BATCHES with it) and re-issue the identical command --\n"
        "the resumable executables continue from their CSV. Records that were in\n"
        "flight are lost, and they are the non-zero ones, so the resumed GMN mean is\n"
        "biased low until new records dilute it; the resume banner reports how many.\n");

    // _Exit, not exit: no atexit handler or stream flush is safe here, and the
    // shell still sees the signal through the 128+n convention.
    std::_Exit(128 + number);
}

} // namespace detail

// Snapshot what the process is doing. Safe to call repeatedly -- callers set a
// coarse context at startup and refine it once the run's settings resolve --
// but never from a signal handler.
inline void set_context(const std::string &text)
{
    const std::size_t length = text.size() < CONTEXT_CAPACITY ? text.size() : CONTEXT_CAPACITY - 1;
    // Publish the length last so a handler firing mid-copy sees either the old
    // text or nothing, never a half-written buffer.
    g_context_length.store(0, std::memory_order_relaxed);
    for (std::size_t i = 0; i < length; ++i)
    {
        g_context[i] = text[i];
    }
    g_context[length] = '\0';
    g_context_length.store(length, std::memory_order_release);
}

// `label` must outlive the process -- pass a string literal. `counter` is read
// at crash time, so it must outlive the run too.
inline void watch(const char *label, const std::atomic<std::uint64_t> &counter)
{
    const std::size_t index = g_counter_count.load(std::memory_order_relaxed);
    if (index >= MAX_COUNTERS)
    {
        return;
    }
    g_counters[index].label = label;
    g_counters[index].value = &counter;
    g_counter_count.store(index + 1, std::memory_order_release);
}

inline void forget_counters()
{
    g_counter_count.store(0, std::memory_order_release);
}

// MIPT_DISABLE_CRASH_HANDLER=1 turns this off, e.g. to let a debugger or a core
// dump see the signal instead. FGMN_DISABLE_CRASH_HANDLER is honoured too, for
// the fgmn.exe flag that predates this header.
inline bool disabled()
{
    for (const char *name : {"MIPT_DISABLE_CRASH_HANDLER", "FGMN_DISABLE_CRASH_HANDLER"})
    {
        const char *value = std::getenv(name);
        if (value != nullptr && value[0] != '\0' && value[0] != '0')
        {
            return true;
        }
    }
    return false;
}

inline void install(const char *program)
{
    if (disabled())
    {
        return;
    }
    g_program = program;
#if !defined(_WIN32)
    std::signal(SIGSEGV, detail::handler);
    std::signal(SIGABRT, detail::handler);
    std::signal(SIGBUS, detail::handler);
    std::signal(SIGILL, detail::handler);
    std::signal(SIGFPE, detail::handler);
#endif
}

} // namespace mipt::util::crash
