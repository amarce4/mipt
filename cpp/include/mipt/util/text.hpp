#pragma once

// Text formatting shared by the command-line front ends: durations for
// progress lines, compact numbers for generated filenames, and quoting for
// CSV fields and re-executed shell commands.
//
// Each of these previously existed as a private copy in several translation
// units. Keeping one copy is what stops the progress lines and generated
// filenames of different executables from drifting apart.

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace mipt::util
{

// Rendering of an elapsed time or an ETA. The styles are not interchangeable:
// each executable's progress output has a fixed look, and callers pick the one
// that matches theirs.
enum class DurationStyle
{
    Compact, // "1h 2m 3s", leading zero units omitted; invalid -> "--"
    Clock,   // "01:02:03";                            invalid -> "--:--:--"
    Padded,  // "01h 02m 03s";                         invalid -> "--h --m --s"
};

inline std::string format_duration(double seconds, DurationStyle style = DurationStyle::Padded)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
    {
        switch (style)
        {
        case DurationStyle::Compact:
            return "--";
        case DurationStyle::Clock:
            return "--:--:--";
        case DurationStyle::Padded:
            break;
        }
        return "--h --m --s";
    }

    const auto total = static_cast<std::uint64_t>(std::llround(seconds));
    const std::uint64_t hours = total / 3600;
    const std::uint64_t minutes = (total % 3600) / 60;
    const std::uint64_t secs = total % 60;

    std::ostringstream out;
    switch (style)
    {
    case DurationStyle::Compact:
        if (hours > 0)
        {
            out << hours << "h ";
        }
        if (hours > 0 || minutes > 0)
        {
            out << minutes << "m ";
        }
        out << secs << 's';
        break;
    case DurationStyle::Clock:
        out << std::setfill('0') << std::setw(2) << hours << ':' << std::setw(2) << minutes << ':' << std::setw(2)
            << secs;
        break;
    case DurationStyle::Padded:
        out << std::setfill('0') << std::setw(2) << hours << "h " << std::setw(2) << minutes << "m " << std::setw(2)
            << secs << 's';
        break;
    }
    return out.str();
}

// Shortest fixed-point spelling of a value, for generated filenames:
// 0.333 -> "0.333", 4.0 -> "4".
inline std::string compact_decimal(double value)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(12) << value;
    std::string text = out.str();
    while (!text.empty() && text.back() == '0')
    {
        text.pop_back();
    }
    if (!text.empty() && text.back() == '.')
    {
        text.pop_back();
    }
    return text == "-0" ? "0" : text;
}

// Round-trippable spelling, for values passed to a re-executed subprocess.
inline std::string precise_decimal(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

// 10000 -> "10k", 2000000 -> "2m", 1234 -> "1234".
inline std::string compact_count(std::uint64_t count)
{
    if (count >= 1000000 && count % 1000000 == 0)
    {
        return std::to_string(count / 1000000) + "m";
    }
    if (count >= 1000 && count % 1000 == 0)
    {
        return std::to_string(count / 1000) + "k";
    }
    return std::to_string(count);
}

inline std::string csv_quote(std::string_view value)
{
    std::string quoted("\"");
    for (char character : value)
    {
        if (character == '"')
        {
            quoted += "\"\"";
        }
        else
        {
            quoted += character;
        }
    }
    quoted += '"';
    return quoted;
}

inline std::string shell_quote(std::string_view value)
{
    std::string quoted("'");
    for (char character : value)
    {
        if (character == '\'')
        {
            quoted += "'\\''";
        }
        else
        {
            quoted += character;
        }
    }
    quoted += '\'';
    return quoted;
}

} // namespace mipt::util
