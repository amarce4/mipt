#pragma once

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>

namespace mipt::env
{
inline bool equals_ignore_case(std::string_view lhs, std::string_view rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        const auto a = static_cast<unsigned char>(lhs[i]);
        const auto b = static_cast<unsigned char>(rhs[i]);
        if ((a >= 'A' && a <= 'Z' ? a + ('a' - 'A') : a) !=
            (b >= 'A' && b <= 'Z' ? b + ('a' - 'A') : b))
        {
            return false;
        }
    }
    return true;
}

inline bool truthy(const char *value)
{
    if (value == nullptr)
    {
        return false;
    }
    const std::string_view token(value);
    return token == "1" || equals_ignore_case(token, "true") ||
           equals_ignore_case(token, "yes") || equals_ignore_case(token, "on");
}

inline bool falsey(const char *value)
{
    if (value == nullptr)
    {
        return false;
    }
    const std::string_view token(value);
    return token == "0" || equals_ignore_case(token, "false") ||
           equals_ignore_case(token, "no") || equals_ignore_case(token, "off");
}

inline bool has_value(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0';
}

inline void set_if_unset(const char *name, const char *value)
{
    if (std::getenv(name) != nullptr)
    {
        return;
    }
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

inline bool boolean(const char *name, bool fallback)
{
    const char *value = std::getenv(name);
    if (truthy(value))
    {
        return true;
    }
    if (falsey(value))
    {
        return false;
    }
    return fallback;
}

inline long integer(const char *name, long fallback, long min_value,
                    long max_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }

    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < min_value ||
        parsed > max_value)
    {
        return fallback;
    }
    return parsed;
}

inline double real(const char *name, double fallback, double min_value,
                   double max_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }

    char *end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed) ||
        parsed < min_value || parsed > max_value)
    {
        return fallback;
    }
    return parsed;
}

inline std::string text(const char *name, const char *fallback = "")
{
    const char *value = std::getenv(name);
    return value == nullptr || *value == '\0' ? std::string(fallback)
                                               : std::string(value);
}

inline bool disables_path(const char *value)
{
    if (value == nullptr)
    {
        return false;
    }
    const std::string_view token(value);
    return falsey(value) || equals_ignore_case(token, "none") ||
           equals_ignore_case(token, "disabled");
}
} // namespace mipt::env
