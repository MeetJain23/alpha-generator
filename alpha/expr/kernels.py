"""numba kernels for every windowed operator.

Scope
-----
Every ``WarmupRule.WINDOW_MINUS_1`` operator is a kernel. Every other operator
stays pure numpy. That line is wider than "the handful where numpy is
pathological" and the reason is the degradation counter, not speed.

A windowed operator has to know, per output cell, how many observations were
actually present in its window. It needs that for ``min_periods`` regardless.
Computing it in numpy means materialising a rolling count plane and comparing
it, which is an extra array of the same size as the data and two extra passes
over it, per node. Inside a kernel the count is already in a register, so the
counter costs one integer comparison and nothing else. The whole point of
scalar degradation counters is to avoid per-node planes, and implementing them
in numpy would reintroduce exactly the planes they exist to avoid.

So: ts_mean, ts_std, ts_rank, ts_min, ts_max, ts_argmax, decay_linear and
correlation are kernels. delta and ts_delay are point lookups with no window
and no count, and they stay numpy.

Compilation flags
-----------------
``cache=True`` on every kernel. Compilation costs roughly half a second per
function, which would dwarf a 50ms evaluation budget in a short-lived worker
process, and the generator is expected to run in exactly that shape.

``fastmath=False``, set explicitly rather than inherited. fastmath permits
reassociation and tells the compiler to assume no NaNs are present. Every
kernel here tests for NaN with ``v == v`` and every one of them depends on NaN
propagating, so fastmath would fold those tests to True and silently void the
entire NaN policy. Not a performance trade: it would produce wrong numbers
that look like right ones.

``error_model="numpy"`` so division follows array semantics rather than
raising. Every division here is guarded anyway; this removes the possibility
of a kernel raising from inside a loop over seven million cells.

Accumulation
------------
Scalar accumulators are float64, output is float32. Accumulators live in
registers, so the wider type costs nothing, and the narrow one would be a real
problem. The naive rolling variance, ``sumsq - sum * sum / m``, is a
difference of two large nearly equal numbers. On price-level data over a
250-day window in float32 it cancels catastrophically, and the result can come
out negative, at which point ``sqrt`` produces a NaN from nowhere. In a system
where NaN is supposed to mean "there was nothing to compute this from", a NaN
manufactured by floating point is worse than a wrong number: it is a wrong
number wearing the uniform of a correct answer.

Negative variance is clamped to zero and counted. The evaluator logs the count
if it is ever non-zero, because in float64 it should not be.

Determinism
-----------
Every kernel is a single forward pass over time whose state at row t depends
only on rows at or before t. Evaluating on ``panel[:k]`` therefore runs an
identical sequence of floating point operations to evaluating on the full
panel and taking the first k rows, so the results agree bit for bit rather
than to a tolerance. ``scripts/check_lookahead.py`` asserts exactly that.

Layout
------
Kernels loop over time on the outside and instruments on the inside, matching
the C-contiguous (n_days x n_instruments) panel layout, so each step reads one
contiguous row. Per-instrument state lives in arrays indexed by instrument.
The evaluator's convention is that nothing here allocates more than one output
plane plus O(n_instruments) or O(window x n_instruments) of state.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numba import njit

KERNEL_FLAGS: Final[dict[str, object]] = {
    "cache": True,
    "fastmath": False,
    "nogil": True,
    "error_model": "numpy",
}


# --------------------------------------------------------------------------
# running-sum kernels
# --------------------------------------------------------------------------


@njit(**KERNEL_FLAGS)
def ts_mean(x, d, min_periods):
    """Rolling mean over present observations.

    Returns ``(out, degraded, considered, clamped)``. The input plane is its
    own ring buffer: the value leaving the window at row t is ``x[t - d]``, so
    no copy of the window is needed.
    """
    n_days, n_inst = x.shape
    out = np.full((n_days, n_inst), np.nan, dtype=np.float32)
    total = np.zeros(n_inst, dtype=np.float64)
    count = np.zeros(n_inst, dtype=np.int64)
    degraded = 0
    considered = 0

    for t in range(n_days):
        for j in range(n_inst):
            v = x[t, j]
            if v == v:
                total[j] += np.float64(v)
                count[j] += 1
            if t >= d:
                w = x[t - d, j]
                if w == w:
                    total[j] -= np.float64(w)
                    count[j] -= 1
            m = count[j]
            if t >= d - 1 and m >= min_periods:
                out[t, j] = np.float32(total[j] / m)
                considered += 1
                if m < d:
                    degraded += 1
    return out, degraded, considered, 0


@njit(**KERNEL_FLAGS)
def ts_std(x, d, min_periods):
    """Rolling sample standard deviation, ddof=1, over present observations.

    Variance is clamped at zero and the clamp is counted. In float64 it should
    never fire; if it does, the accumulators have drifted far enough that the
    caller needs to know rather than receive a NaN with no explanation.
    """
    n_days, n_inst = x.shape
    out = np.full((n_days, n_inst), np.nan, dtype=np.float32)
    total = np.zeros(n_inst, dtype=np.float64)
    total_sq = np.zeros(n_inst, dtype=np.float64)
    count = np.zeros(n_inst, dtype=np.int64)
    degraded = 0
    considered = 0
    clamped = 0

    for t in range(n_days):
        for j in range(n_inst):
            v = x[t, j]
            if v == v:
                # Widen before squaring. A float32 product of a price-level
                # value is already short of the precision the accumulator
                # was widened to provide, and storing it in a float64 slot
                # does not put back the bits the multiply threw away.
                vd = np.float64(v)
                total[j] += vd
                total_sq[j] += vd * vd
                count[j] += 1
            if t >= d:
                w = x[t - d, j]
                if w == w:
                    wd = np.float64(w)
                    total[j] -= wd
                    total_sq[j] -= wd * wd
                    count[j] -= 1
            m = count[j]
            if t >= d - 1 and m >= min_periods and m >= 2:
                mean = total[j] / m
                variance = (total_sq[j] - mean * total[j]) / (m - 1)
                if variance < 0.0:
                    variance = 0.0
                    clamped += 1
                out[t, j] = np.float32(np.sqrt(variance))
                considered += 1
                if m < d:
                    degraded += 1
    return out, degraded, considered, clamped


@njit(**KERNEL_FLAGS)
def decay_linear(x, d, min_periods):
    """Linearly weighted mean, weights d, d-1, ..., 1, renormalised.

    The weights are rescaled over the observations actually present, never
    over the full window. Treating a missing observation as zero weight
    without rescaling would shrink the result in proportion to how much data
    an instrument is missing, and missingness concentrates in halted and
    thinly traded names, so the damping would land on one group of
    instruments and the search would find it as a liquidity factor.

    Two recurrences, both O(1) per cell. With
    ``W(t) = sum_i (d - i) * x[t - i]`` and ``S(t) = sum_i x[t - i]`` over the
    window ending at t, ``W(t + 1) = d * x[t + 1] + W(t) - S(t)``, where S is
    the sum before the departing observation is evicted. The weight total V
    follows the same recurrence with 1 in place of x.
    """
    n_days, n_inst = x.shape
    out = np.full((n_days, n_inst), np.nan, dtype=np.float32)
    weighted = np.zeros(n_inst, dtype=np.float64)
    weight_total = np.zeros(n_inst, dtype=np.float64)
    total = np.zeros(n_inst, dtype=np.float64)
    count = np.zeros(n_inst, dtype=np.int64)
    degraded = 0
    considered = 0

    for t in range(n_days):
        for j in range(n_inst):
            v = x[t, j]
            present = 1.0 if v == v else 0.0
            value = np.float64(v) if v == v else 0.0

            weighted[j] += d * value - total[j]
            weight_total[j] += d * present - count[j]

            total[j] += value
            if present > 0.0:
                count[j] += 1
            if t >= d:
                w = x[t - d, j]
                if w == w:
                    total[j] -= np.float64(w)
                    count[j] -= 1

            m = count[j]
            if t >= d - 1 and m >= min_periods and weight_total[j] > 0.0:
                out[t, j] = np.float32(weighted[j] / weight_total[j])
                considered += 1
                if m < d:
                    degraded += 1
    return out, degraded, considered, 0


@njit(**KERNEL_FLAGS)
def correlation(x, y, d, min_periods):
    """Rolling Pearson correlation over pairwise-complete observations.

    A day enters the window only if both inputs are present on it, so the
    moments are accumulated over exactly the days that carry information about
    the relationship. Either variance vanishing gives NaN rather than zero:
    there is no correlation to report with a constant series, and zero would
    claim there was and that it was none.
    """
    n_days, n_inst = x.shape
    out = np.full((n_days, n_inst), np.nan, dtype=np.float32)
    sx = np.zeros(n_inst, dtype=np.float64)
    sy = np.zeros(n_inst, dtype=np.float64)
    sxx = np.zeros(n_inst, dtype=np.float64)
    syy = np.zeros(n_inst, dtype=np.float64)
    sxy = np.zeros(n_inst, dtype=np.float64)
    count = np.zeros(n_inst, dtype=np.int64)
    degraded = 0
    considered = 0
    clamped = 0

    for t in range(n_days):
        for j in range(n_inst):
            a = x[t, j]
            b = y[t, j]
            if a == a and b == b:
                # Widen before forming any product, for the same reason as
                # ts_std: the multiply is where the precision is lost.
                ad = np.float64(a)
                bd = np.float64(b)
                sx[j] += ad
                sy[j] += bd
                sxx[j] += ad * ad
                syy[j] += bd * bd
                sxy[j] += ad * bd
                count[j] += 1
            if t >= d:
                a_old = x[t - d, j]
                b_old = y[t - d, j]
                if a_old == a_old and b_old == b_old:
                    ao = np.float64(a_old)
                    bo = np.float64(b_old)
                    sx[j] -= ao
                    sy[j] -= bo
                    sxx[j] -= ao * ao
                    syy[j] -= bo * bo
                    sxy[j] -= ao * bo
                    count[j] -= 1

            m = count[j]
            if t >= d - 1 and m >= min_periods and m >= 2:
                mean_x = sx[j] / m
                mean_y = sy[j] / m
                var_x = sxx[j] - mean_x * sx[j]
                var_y = syy[j] - mean_y * sy[j]
                if var_x < 0.0:
                    var_x = 0.0
                    clamped += 1
                if var_y < 0.0:
                    var_y = 0.0
                    clamped += 1
                if var_x > 0.0 and var_y > 0.0:
                    cov = sxy[j] - mean_x * sy[j]
                    r = cov / np.sqrt(var_x * var_y)
                    if r > 1.0:
                        r = 1.0
                    elif r < -1.0:
                        r = -1.0
                    out[t, j] = np.float32(r)
                    considered += 1
                    if m < d:
                        degraded += 1
    return out, degraded, considered, clamped


# --------------------------------------------------------------------------
# monotonic deque kernels
# --------------------------------------------------------------------------
#
# ts_min, ts_max and ts_argmax share one structure: a per-instrument deque of
# row indices whose values are monotonic, so the extreme of the window is
# always at the front. Each index is pushed once and popped once, which makes
# the whole pass O(n_days) per instrument rather than O(n_days * d).
#
# The deques live in one (d x n_instruments) int32 buffer with head and tail
# counters per instrument, so the inner loop over instruments still walks
# contiguous memory for the data itself.


@njit(**KERNEL_FLAGS)
def _extremum(x, d, min_periods, want_max, want_index):
    """Shared driver for ts_min, ts_max and ts_argmax.

    ``want_index`` returns rows elapsed since the extreme rather than its
    value. That index counts calendar positions inside the window, not present
    observations, so a maximum five rows back is 5 whether or not the rows
    between it and now were traded. Counting present observations instead
    would make the same number mean different elapsed times in different
    cells, and the output would stop being comparable across the
    cross-section.
    """
    n_days, n_inst = x.shape
    out = np.full((n_days, n_inst), np.nan, dtype=np.float32)
    ring = np.zeros((d, n_inst), dtype=np.int32)
    head = np.zeros(n_inst, dtype=np.int64)
    tail = np.zeros(n_inst, dtype=np.int64)
    count = np.zeros(n_inst, dtype=np.int64)
    degraded = 0
    considered = 0

    for t in range(n_days):
        for j in range(n_inst):
            # Expire first. After a push the deque can hold at most d
            # indices, but expiring afterwards would let it briefly hold
            # d + 1 and wrap the ring onto its own front.
            while tail[j] > head[j] and ring[head[j] % d, j] <= t - d:
                head[j] += 1

            v = x[t, j]
            if v == v:
                # Drop indices that this value dominates: they can never be
                # the extreme again while it is in the window.
                while tail[j] > head[j]:
                    last = ring[(tail[j] - 1) % d, j]
                    prior = x[last, j]
                    if want_max:
                        dominated = prior <= v
                    else:
                        dominated = prior >= v
                    if not dominated:
                        break
                    tail[j] -= 1
                ring[tail[j] % d, j] = t
                tail[j] += 1
                count[j] += 1
            if t >= d:
                w = x[t - d, j]
                if w == w:
                    count[j] -= 1

            m = count[j]
            if t >= d - 1 and m >= min_periods and tail[j] > head[j]:
                front = ring[head[j] % d, j]
                if want_index:
                    out[t, j] = np.float32(t - front)
                else:
                    out[t, j] = x[front, j]
                considered += 1
                if m < d:
                    degraded += 1
    return out, degraded, considered, 0


@njit(**KERNEL_FLAGS)
def ts_max(x, d, min_periods):
    """Rolling maximum over present observations."""
    return _extremum(x, d, min_periods, True, False)


@njit(**KERNEL_FLAGS)
def ts_min(x, d, min_periods):
    """Rolling minimum over present observations."""
    return _extremum(x, d, min_periods, False, False)


@njit(**KERNEL_FLAGS)
def ts_argmax(x, d, min_periods):
    """Rows elapsed since the window maximum, 0 meaning today."""
    return _extremum(x, d, min_periods, True, True)


# --------------------------------------------------------------------------
# rank
# --------------------------------------------------------------------------


@njit(**KERNEL_FLAGS)
def ts_rank(x, d, min_periods):
    """Rank of x[t] among the present observations of its trailing window.

    Normalised by the present count m, never by the window length d. Dividing
    by d would cap a degraded window's output at m/d however extreme the value
    was, and since missing observations concentrate in halted and thinly
    traded names, the damping would land on one group of instruments and
    manufacture a liquidity factor.

    Ties take the average position, so the result is
    ``(less + (equal + 1) / 2) / m`` and lies in (0, 1].

    This is the one kernel that is O(d) per cell rather than O(1). A sliding
    window rank has no constant-time update, because removing one observation
    and adding another reorders the rest arbitrarily. The input is transposed
    by the caller so each instrument's history is contiguous, which keeps the
    inner scan inside one or two cache lines for the shorter windows.
    """
    n_inst, n_days = x.shape
    out = np.full((n_inst, n_days), np.nan, dtype=np.float32)
    degraded = 0
    considered = 0

    for j in range(n_inst):
        for t in range(d - 1, n_days):
            v = x[j, t]
            if v != v:
                continue
            less = 0
            equal = 0
            m = 0
            for k in range(t - d + 1, t + 1):
                w = x[j, k]
                if w != w:
                    continue
                m += 1
                if w < v:
                    less += 1
                elif w == v:
                    equal += 1
            if m >= min_periods:
                out[j, t] = np.float32((less + (equal + 1) * 0.5) / m)
                considered += 1
                if m < d:
                    degraded += 1
    return out, degraded, considered, 0
