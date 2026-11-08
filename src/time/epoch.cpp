// Epoch: the parts compiled out of line (only the system-clock read).

#include "ballistic/time/epoch.hpp"
#include <chrono>

namespace ballistic {
namespace time {

// The current time from the system clock, read as Unix time (seconds since
// 1970-01-01 00:00:00 UTC, leap seconds not counted). Accurate only to the
// system clock's own synchronization.
Epoch Epoch::now() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    double unix_time = std::chrono::duration<double>(duration).count();
    return Epoch::from_unix(unix_time);
}

} // namespace time
} // namespace ballistic
