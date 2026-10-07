//! The tail pre-fault helper of the checker pool. Not a Go port: Go zeroes
//! new memory on the goroutine that first touches it, too.
//!
//! PERF (tailfault1): in a check group (`program.rs`
//! `start_checker_group_do`) one checker often has much more work than the
//! others. While it is the last checker with work, the kernel zeroes each
//! new page that it touches first on the critical path (perfmeas1, cup2:
//! vue-macros 120 ms, 10% of the wall; blueprint 118 ms, 7.5%), and the
//! cores of the other checkers are free.
//!
//! When one checker is left with work in a group, the `prefault` thread
//! keeps a small reserve of touched free memory in that checker's jemalloc
//! arena. It reads the arena's dirty pages (`stats.arenas.<i>.pdirty`: free
//! pages that were touched before, which jemalloc hands out first) every
//! `POLL`. Below `RESERVE_LOW` it allocates `FILL` bytes in that arena,
//! writes one byte per 4 KiB page and frees them. The checker's next
//! allocations then find pages that are already there.
//!
//! - The arena keeps its dirty pages while it is served
//!   (`arena.<i>.dirty_decay_ms` -1; the old value comes back after). With
//!   the default decay, jemalloc purged a fresh reserve within 1 to 2 ms
//!   (prefault1).
//! - Half of a fill is in 128 KiB blocks: jemalloc splits a free extent only
//!   for a request of at least 1/64 of it (`opt.lg_extent_max_active_fit`
//!   6), and freed neighbor blocks merge.
//! - A fill holds all its blocks before it frees them, so the first blocks
//!   take the dirty pages that are there and the rest are new pages.
//! - Not calloc: jemalloc memsets a zeroed request that dirty memory serves.
//! - Allocations of 8 MiB or more go to the oversize arena, which the helper
//!   does not serve.
//!
//! Only in one-shot runs (not `--lsp`, `--api` or watch) with jemalloc on
//! Linux and 2 or more CPUs, and for groups of 2 to 64 checkers. An
//! always-on helper (prefault1 Part B) made runs whose work is spread over
//! the checkers slower. A larger reserve (RSS / 64, 2 to 8 MiB, filled to
//! twice that) gained no more and cost 2% to 3% of peak RSS.
//! `GOPORT_TAILFAULT=0` turns the helper off. `GOPORT_TAILFAULT_DEBUG=1`
//! prints, on stderr, each group job's time, minor faults and CPU ticks,
//! and what the helper did (with `GOPORT_TAILFAULT=0` too, without a
//! helper).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// No arena (`TARGET` and the slots of `GroupTail::arenas`).
const NONE: u32 = u32::MAX;

/// The helper fills the served arena when it has less than this in dirty
/// pages.
const RESERVE_LOW: usize = 1 << 20;

/// The bytes of one fill: half in `SMALL_BLOCK` blocks, half in one
/// `BIG_BLOCK`.
const FILL: usize = 2 << 20;
const SMALL_BLOCK: usize = 128 << 10;
const BIG_BLOCK: usize = 1 << 20;

/// The wait between two reads of the served arena.
const POLL: Duration = Duration::from_micros(500);

/// The tail state of one check group: which checkers still run their job,
/// and the jemalloc arena of each. The group's jobs share it.
pub struct GroupTail {
    /// Bit `i` is set while checker `i` has not ended its job.
    running: AtomicU64,
    /// The arena of each checker's thread, `NONE` until its job starts.
    arenas: Box<[AtomicU32]>,
}

/// A checker's part of a group (`GroupTail::job`). It ends when dropped,
/// also after a panic.
pub struct TailJob<'a> {
    tail: &'a GroupTail,
    index: usize,
    /// With `GOPORT_TAILFAULT_DEBUG`: the start time and `thread_stat`.
    debug: Option<(Instant, [u64; 3])>,
}

impl GroupTail {
    /// The tail state of a group of `count` checkers, or None when the
    /// helper is off (and not debugged) or the group has 1 or more than 64
    /// checkers.
    pub fn new(count: usize) -> Option<Arc<GroupTail>> {
        if !(2..=64).contains(&count) || !(enabled() || debug()) {
            return None;
        }
        Some(Arc::new(GroupTail {
            running: AtomicU64::new(u64::MAX >> (64 - count)),
            arenas: (0..count).map(|_| AtomicU32::new(NONE)).collect(),
        }))
    }

    /// Starts checker `index`'s part of the group on its thread.
    pub fn job(&self, index: usize) -> TailJob<'_> {
        let arena = thread_arena().unwrap_or(NONE);
        self.arenas[index].store(arena, Ordering::Release);
        // The others may have ended before this job started.
        if self.running.load(Ordering::Acquire) == 1 << index {
            serve(arena);
        }
        TailJob {
            tail: self,
            index,
            debug: debug().then(|| (Instant::now(), thread_stat())),
        }
    }
}

impl Drop for TailJob<'_> {
    fn drop(&mut self) {
        let bit = 1u64 << self.index;
        let left = self.tail.running.fetch_and(!bit, Ordering::AcqRel) & !bit;
        if let Some((start, before)) = self.debug {
            let now = thread_stat();
            eprintln!(
                "tailfault: checker {} job {:.1} ms, {} minor faults, utime {} stime {} ticks, {} left",
                self.index,
                start.elapsed().as_secs_f64() * 1e3,
                now[0].saturating_sub(before[0]),
                now[1].saturating_sub(before[1]),
                now[2].saturating_sub(before[2]),
                left.count_ones()
            );
        }
        match last_left(left) {
            // A job that has not started has no arena yet: `job` serves it.
            Some(last) => serve(self.tail.arenas[last].load(Ordering::Acquire)),
            None if left == 0 => stop(self.tail.arenas[self.index].load(Ordering::Acquire)),
            None => {}
        }
    }
}

/// The index of the one checker in `left` (a `GroupTail::running` mask), or
/// None when 0 or 2 or more are left.
fn last_left(left: u64) -> Option<usize> {
    (left.count_ones() == 1).then(|| left.trailing_zeros() as usize)
}

/// True when the helper may run: a one-shot run with jemalloc on Linux, 2
/// or more CPUs, and `GOPORT_TAILFAULT` is not `0`. Read once.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(all(feature = "jemalloc", target_os = "linux"))
            && std::env::var_os("GOPORT_TAILFAULT").is_none_or(|v| v != "0")
            && !crate::thp_guard::long_running()
            && crate::gostd::runtime::gomaxprocs() >= 2
    })
}

/// `GOPORT_TAILFAULT_DEBUG` is set and not `0`. Read once.
fn debug() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GOPORT_TAILFAULT_DEBUG").is_some_and(|v| v != "0"))
}

/// The arena that the helper serves, or `NONE`.
static TARGET: AtomicU32 = AtomicU32::new(NONE);

/// Makes the helper serve `arena`. Starts the helper on first use.
fn serve(arena: u32) {
    if arena == NONE || !enabled() {
        return;
    }
    TARGET.store(arena, Ordering::Release);
    if let Some(helper) = helper() {
        helper.unpark();
    }
}

/// Stops the helper when it serves `arena`: the last checker of a group
/// ended. The tail of another group keeps it.
fn stop(arena: u32) {
    let _ = TARGET.compare_exchange(arena, NONE, Ordering::AcqRel, Ordering::Relaxed);
}

/// The helper thread, started on first use. None when it cannot start.
fn helper() -> Option<&'static std::thread::Thread> {
    static HELPER: OnceLock<Option<std::thread::Thread>> = OnceLock::new();
    HELPER
        .get_or_init(|| {
            std::thread::Builder::new()
                .name("prefault".into())
                .stack_size(256 << 10)
                .spawn(helper_loop)
                .ok()
                .map(|handle| handle.thread().clone())
        })
        .as_ref()
}

/// The helper: serves `TARGET` while it is set, else parks.
#[cfg(all(feature = "jemalloc", target_os = "linux"))]
fn helper_loop() {
    use tikv_jemalloc_ctl::{Access, AsName};
    let pdirty = |arena: u32| -> Option<usize> {
        tikv_jemalloc_ctl::epoch::advance().ok()?;
        format!("stats.arenas.{arena}.pdirty\0")
            .as_bytes()
            .name()
            .read()
            .ok()
    };
    let decay_name = |arena: u32| format!("arena.{arena}.dirty_decay_ms\0");
    // The arena served now and its decay before.
    let mut served: Option<(u32, isize)> = None;
    let (mut since, mut polls, mut fills) = (Instant::now(), 0u64, 0u64);
    loop {
        let target = TARGET.load(Ordering::Acquire);
        if served.map(|(arena, _)| arena) != Some(target) {
            if let Some((arena, decay)) = served.take() {
                let _ = decay_name(arena).as_bytes().name().write(decay);
                if debug() {
                    eprintln!(
                        "tailfault: helper served arena {arena} {:.1} ms: {polls} polls, {fills} fills, {} KiB",
                        since.elapsed().as_secs_f64() * 1e3,
                        fills * (FILL as u64 >> 10)
                    );
                }
            }
            if target == NONE {
                std::thread::park();
                continue;
            }
            // This thread's allocations go to the served arena.
            let _ = b"thread.arena\0".name().write(target);
            let name = decay_name(target);
            let decay = name.as_bytes().name().read().unwrap_or(10_000isize);
            let _ = name.as_bytes().name().write(-1isize);
            served = Some((target, decay));
            (since, polls, fills) = (Instant::now(), 0, 0);
        }
        polls += 1;
        if pdirty(target).is_none_or(|pages| pages * 4096 >= RESERVE_LOW) {
            std::thread::park_timeout(POLL);
            continue;
        }
        fill();
        fills += 1;
    }
}

/// Without jemalloc on Linux `enabled` is false, so no helper starts.
#[cfg(not(all(feature = "jemalloc", target_os = "linux")))]
fn helper_loop() {}

/// Allocates `FILL` bytes in this thread's arena, writes one byte per 4 KiB
/// page and frees them all.
fn fill() {
    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(FILL / 2 / SMALL_BLOCK + 1);
    let mut done = 0;
    while done < FILL {
        let size = if done < FILL / 2 {
            SMALL_BLOCK
        } else {
            BIG_BLOCK
        };
        let mut block: Vec<u8> = Vec::with_capacity(size);
        for page in block.spare_capacity_mut().iter_mut().step_by(4096) {
            page.write(1);
        }
        // The writes stay: the block escapes before it is freed.
        std::hint::black_box(&mut block);
        blocks.push(block);
        done += size;
    }
}

/// This thread's jemalloc arena (`thread.arena`), or None.
fn thread_arena() -> Option<u32> {
    #[cfg(all(feature = "jemalloc", not(windows)))]
    {
        use tikv_jemalloc_ctl::{Access, AsName};
        b"thread.arena\0".name().read().ok()
    }
    #[cfg(not(all(feature = "jemalloc", not(windows))))]
    None
}

/// This thread's minor faults and user and system time in clock ticks
/// (`/proc/thread-self/stat` fields 10, 14 and 15), or zeros. Debug output
/// only.
fn thread_stat() -> [u64; 3] {
    let text = std::fs::read_to_string("/proc/thread-self/stat").unwrap_or_default();
    // Field 3 (the state) is the first after the name, which ends at the
    // last ')'.
    let fields: Vec<u64> = text
        .rfind(')')
        .map(|end| {
            text[end + 1..]
                .split_whitespace()
                .map(|field| field.parse().unwrap_or(0))
                .collect()
        })
        .unwrap_or_default();
    let field = |n: usize| fields.get(n - 3).copied().unwrap_or(0);
    [field(10), field(14), field(15)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The helper serves a checker once it is the only one left in its
    /// group, and stops when none is left.
    #[test]
    fn last_left_names_the_one_checker_left() {
        assert_eq!(last_left(0b1011), None);
        assert_eq!(last_left(0b1000), Some(3));
        assert_eq!(last_left(1), Some(0));
        assert_eq!(last_left(1 << 63), Some(63));
        assert_eq!(last_left(0), None);
    }
}
