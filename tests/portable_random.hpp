#pragma once

// Random distributions that give the same numbers with every standard library.
//
// std::mt19937 is specified exactly by the C++ standard, but the
// distributions built on it (std::normal_distribution, ...) are not: GCC's
// libstdc++, LLVM's libc++ and MSVC's library each turn the same engine
// output into different numbers. Tests that add noise with them therefore see
// different data on Linux, macOS and Windows, and a bound measured on one
// fails on another. These classes reproduce libstdc++'s algorithms exactly, so
// every platform draws the numbers the tests were measured with on Linux
// (checked bit for bit against libstdc++ in test_portable_random).

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

namespace ballistic::test {

// libstdc++'s std::generate_canonical<double, 53>(std::mt19937&): two 32-bit
// draws combined as (g0 + g1 * 2^32) / 2^64, kept below 1.
inline double canonical_double(std::mt19937& g) {
    constexpr double r = 4294967296.0;   // 2^32: the range of std::mt19937
    const double g0 = static_cast<double>(g());
    const double g1 = static_cast<double>(g());
    double ret = (g0 + g1 * r) / (r * r);
    if (ret >= 1.0) ret = std::nextafter(1.0, 0.0);
    return ret;
}

// std::uniform_real_distribution<double>(a, b), libstdc++'s algorithm.
class UniformReal {
public:
    explicit UniformReal(double a = 0.0, double b = 1.0) : a_(a), b_(b) {}
    double operator()(std::mt19937& g) { return a_ + (b_ - a_) * canonical_double(g); }

private:
    double a_, b_;
};

// std::normal_distribution<double>(mean, stddev), libstdc++'s algorithm: the
// Marsaglia polar method, keeping the second value of each pair for the next
// call (so one object gives the same sequence as one std object).
class Normal {
public:
    explicit Normal(double mean = 0.0, double stddev = 1.0) : mean_(mean), stddev_(stddev) {}
    double operator()(std::mt19937& g) {
        double ret;
        if (saved_available_) {
            saved_available_ = false;
            ret = saved_;
        } else {
            double x, y, r2;
            do {
                x = 2.0 * canonical_double(g) - 1.0;
                y = 2.0 * canonical_double(g) - 1.0;
                r2 = x * x + y * y;
            } while (r2 > 1.0 || r2 == 0.0);
            const double mult = std::sqrt(-2.0 * std::log(r2) / r2);
            saved_ = x * mult;
            saved_available_ = true;
            ret = y * mult;
        }
        return ret * stddev_ + mean_;
    }

private:
    double mean_, stddev_;
    double saved_ = 0.0;
    bool saved_available_ = false;
};

}  // namespace ballistic::test
