"""EVIDENCE — how much can we trust the other five sections?

This section does not test the strategy. It tests the other verdicts, by
asking what sample stands behind them. Its output is a weight, not a
finding about the edge.

THE DENOMINATOR PROBLEM
-----------------------
A blotter row is not a bet. Before any significance statement can be made,
one question has to be answered from the records: how many INDEPENDENT
things did we observe? Three corrections, applied in this order, because
each one decides what the next is allowed to do:

1. Trades -> round trips. Every position is opened and closed, so half the
   rows are the mirror of the other half. Mechanical, a factor of 2.

2. Round trips -> delivery days. Products within one delivery day are NOT
   independent: one weather run revises the whole day, which is the
   revision family's own pitch. Measured here as the intraclass
   correlation of product-level P&L within day; the design effect
   1 + (m-1)*ICC says how many bets a day is really worth.

3. Delivery days -> effectively participating days. Even independent days
   do not contribute equally when the P&L is concentrated. The
   participation ratio (sum|x|)^2 / sum(x^2) counts how many days actually
   carry the total: it equals N for evenly-spread P&L and 1 when a single
   day is everything.

Across days the series is close to serially independent (lag-1
autocorrelation is reported so this is checked, not assumed), so no block
correction is needed at the day level. Within days it is not. That
asymmetry is the whole finding: the autocorrelation correction is nearly
free here, the clustering correction is worth 2-3x.

EFFECTIVE SAMPLE SIZE IS NOT THE BOOTSTRAP
------------------------------------------
They are sequential, not alternative. The bootstrap cannot discover the
unit of independence — it resamples whatever it is handed, and returns a
confident interval either way. Resampling product bets rather than days
gives windfall2 a t of 24 instead of 7. So:

    autocorrelation + ICC  ->  establishes the unit
    bootstrap at that unit ->  the interval

What the bootstrap adds that the sample-size work cannot: it drops the
normality assumption. With daily kurtosis between 4 and 134, the textbook
Sharpe standard error is not usable; the bootstrap does not care.

Thresholds are fixed here, before any submission is looked at, and apply
unchanged to all seven.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from report import Finding, SectionResult

# --- verdict thresholds, fixed in advance ----------------------------------
MIN_EFF_DAYS = 50.0    # effectively participating days for a confident verdict
DROP_MAX = 30          # how far the leave-out curve is computed
T_FRAGILE = 2.0        # t below which the result stops being significant
DROP_FRAGILE = 10      # best days it must take to get there, or evidence is thin
SR_CI_FLOOR = 1.0      # bootstrap 95% lower bound on annualised Sharpe
DEFF_WARN = 2.0        # design effect above which naive counting misleads
N_BOOT = 4000
BOOT_SEED = 7
BLOCK_MEAN = 5         # mean block length (days) for the stationary bootstrap
RHO_LAGS = 10          # lags carried into the Lo annualisation factor
PERIODS = 365          # daily -> annual
Z95 = 1.959964


@dataclass
class Weight:
    """The sample behind one submission's numbers."""
    submission: str
    trades: int
    round_trips: int
    product_bets: int
    delivery_days: int          # active days: the independent unit
    calendar_days: int
    products_per_day: float
    icc: float                  # intraclass corr of product P&L within day
    design_effect: float        # 1 + (m-1)*ICC
    rho1: float                 # lag-1 autocorrelation of daily P&L
    eff_days: float             # participation ratio
    trades_per_day: float
    sharpe: float
    t_product: float            # counting every product bet as independent
    t_day: float                # counting every delivery day as independent
    ci_lo: float                # day-level bootstrap, 95%
    ci_hi: float
    blk_lo: float               # stationary block bootstrap, 95%
    blk_hi: float
    drop_curve: list[float]     # t after deleting the k best days, k=0..DROP_MAX
    drop_to_t2: int | None      # best days to delete before t < T_FRAGILE
    skew: float
    kurtosis: float             # not excess
    rho: list[float]            # daily P&L autocorrelation, lags 1..RHO_LAGS
    eta: float                  # Lo annualisation factor, vs sqrt(PERIODS)

    @property
    def inflation(self) -> float:
        """How overconfident the naive unit would have made you."""
        return self.t_product / self.t_day if self.t_day else np.nan


def _annualised_sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(365.0)) if s > 0 else 0.0


def _icc_within_day(product_pnl: pd.Series) -> tuple[float, float]:
    """One-way random-effects ICC of product P&L grouped by delivery day.

    Returns (icc, mean products per day). Negative ICC is possible and
    means "no more alike than chance"; it is clipped at zero where it
    feeds the design effect, but reported raw.
    """
    day = product_pnl.index.normalize()
    g = product_pnl.groupby(day)
    m = g.size()
    if len(m) < 2 or len(product_pnl) <= len(m):
        return 0.0, float(m.mean()) if len(m) else 0.0
    mbar = float(m.mean())
    grand = product_pnl.mean()
    ms_between = (g.mean().sub(grand).pow(2) * m).sum() / (len(m) - 1)
    ms_within = (g.apply(lambda s: s.sub(s.mean()).pow(2).sum()).sum()
                 / (len(product_pnl) - len(m)))
    denom = ms_between + (mbar - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0.0
    return float(icc), mbar


def _leave_out_curve(v: np.ndarray, k_max: int) -> tuple[list[float], int | None]:
    """t-statistic after deleting the k best days, for k = 0..k_max.

    The participation ratio says how concentrated the P&L is; this says what
    that concentration costs. A result that needs only a handful of days
    removed to stop being significant is resting on those days, however long
    the calendar window was.

    Read in one direction only. A curve that RISES as the best days come out
    means the edge is relentless rather than spiky — which is a strength for
    a market-making strategy and a red flag for one that already failed
    Lineage, so it is reported but never used to upgrade a verdict.
    """
    order = np.argsort(-v)
    curve, first = [], None
    for k in range(k_max + 1):
        keep = np.delete(v, order[:k])
        sd = keep.std(ddof=1)
        t = keep.mean() / (sd / np.sqrt(len(keep))) if sd > 0 else 0.0
        curve.append(float(t))
        if first is None and t < T_FRAGILE:
            first = k
    return curve, first


def _lo_eta(rho: list[float], q: int = PERIODS) -> float:
    """Lo (2002) scaling factor for annualising an autocorrelated Sharpe.

    Independent returns annualise by sqrt(q). When returns are serially
    correlated they do not: the variance of the q-period sum grows faster
    (positive rho) or slower (negative rho) than q, so the honest factor is

        eta(q) = q / sqrt( q + 2 * sum_{k=1..q-1} (q-k) * rho_k )

    Positive autocorrelation therefore DEFLATES the annualised Sharpe.
    Only the first RHO_LAGS autocorrelations are estimable from a sample
    this size; the rest are taken as zero, which is the conventional
    truncation and is stated on the page rather than hidden.
    """
    s = float(q) + 2.0 * sum((q - k) * r for k, r in enumerate(rho, start=1))
    return float(q / np.sqrt(s)) if s > 0 else float(np.sqrt(q))


def _sharpe_se(sr_period: float, g3: float, g4: float, n: float) -> float:
    """Standard error of a per-period Sharpe under non-normal returns.

    Lo (2002) / Mertens: Var(SR) = (1 + SR^2/2 - g3*SR + (g4-3)*SR^2/4) / n.
    The skew and kurtosis terms matter here — daily kurtosis reaches 134 —
    and n is whatever the sample-size work says it really is.
    """
    if n <= 1:
        return float("nan")
    v = (1.0 + sr_period ** 2 / 2.0 - g3 * sr_period
         + (g4 - 3.0) * sr_period ** 2 / 4.0) / n
    return float(np.sqrt(v)) if v > 0 else float("nan")


def deflation(w: "Weight", claimed: float | None = None) -> list[dict]:
    """The annualised Sharpe as each correction is applied, with its interval.

    One strategy, four readings of the same trades:

      1. what was claimed
      2. rebuilt from the records, every product bet counted as independent
      3. re-annualised for the autocorrelation of daily P&L (Lo)
      4. re-intervalled for the sample that actually exists

    The point estimate moves once (step 3, the annualisation factor); the
    interval widens twice (steps 3 and 4, the denominator). Both matter and
    they are different things.
    """
    sr_d = w.sharpe / np.sqrt(PERIODS)          # back out the per-day Sharpe
    se_d = lambda n: _sharpe_se(sr_d, w.skew, w.kurtosis, n)

    def band(point: float, scale: float, n: float) -> tuple[float, float]:
        se = se_d(n)
        if not np.isfinite(se):
            return float("nan"), float("nan")
        return point - Z95 * scale * se, point + Z95 * scale * se

    naive = sr_d * np.sqrt(PERIODS)
    adj = sr_d * w.eta

    rows = []
    if claimed is not None:
        rows.append({"label": "claimed", "sharpe": float(claimed),
                     "lo": None, "hi": None,
                     "note": "as submitted"})
    lo2, hi2 = band(naive, np.sqrt(PERIODS), w.product_bets)
    rows.append({"label": "every product bet independent", "sharpe": naive,
                 "lo": lo2, "hi": hi2,
                 "note": f"n = {w.product_bets:,} bets"})
    lo3, hi3 = band(adj, w.eta, w.delivery_days)
    rows.append({"label": "autocorrelation-adjusted", "sharpe": adj,
                 "lo": lo3, "hi": hi3,
                 "note": f"Lo factor {w.eta:.1f} vs √{PERIODS} = "
                         f"{np.sqrt(PERIODS):.1f}; n = {w.delivery_days:,} days"})
    lo4, hi4 = band(adj, w.eta, w.eff_days)
    rows.append({"label": "effective-sample-adjusted", "sharpe": adj,
                 "lo": lo4, "hi": hi4,
                 "note": f"n = {w.eff_days:.0f} effective days"})
    return rows


def _participation_ratio(v: np.ndarray) -> float:
    """How many days effectively carry the total P&L.

    (sum|x|)^2 / sum(x^2): equals N when every day contributes the same
    magnitude, and 1 when one day is the whole result.
    """
    ss = float((v ** 2).sum())
    return float(np.abs(v).sum() ** 2 / ss) if ss > 0 else 0.0


def _stationary_bootstrap(v: np.ndarray, n_boot: int, mean_block: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: geometric blocks, wrapped.

    Allows for serial dependence the iid bootstrap would miss. Here it is
    a check rather than a correction — with lag-1 autocorrelation near
    zero it should land on top of the iid interval, and saying so is the
    point.
    """
    n = len(v)
    p = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    jumps = rng.random((n_boot, n)) < p
    idx = np.empty((n_boot, n), dtype=int)
    idx[:, 0] = starts[:, 0]
    for j in range(1, n):
        idx[:, j] = np.where(jumps[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return v[idx]


def measure(submission: str, blotter: pd.DataFrame, product_pnl: pd.Series,
            daily: pd.Series) -> Weight:
    """All the sample-weight diagnostics for one submission.

    `product_pnl` and `daily` come from repricer.product_pnl / daily_pnl,
    so this module never re-derives P&L — it only counts what stands
    behind it.
    """
    from lineage import mark_open_close

    v = daily.to_numpy(dtype=float)
    active = v[v != 0]
    n_active = int((v != 0).sum())

    icc, mbar = _icc_within_day(product_pnl)
    deff = 1.0 + (mbar - 1.0) * max(icc, 0.0)

    pv = product_pnl.to_numpy(dtype=float)
    t_prod = (pv.mean() / (pv.std(ddof=1) / np.sqrt(len(pv)))
              if len(pv) > 1 and pv.std(ddof=1) > 0 else np.nan)
    t_day = (v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
             if len(v) > 1 and v.std(ddof=1) > 0 else np.nan)

    rng = np.random.default_rng(BOOT_SEED)
    iid = rng.choice(v, (N_BOOT, len(v)), replace=True)
    sr_iid = np.sort([_annualised_sharpe(r) for r in iid])
    blk = _stationary_bootstrap(v, N_BOOT // 4, BLOCK_MEAN, rng)
    sr_blk = np.sort([_annualised_sharpe(r) for r in blk])

    curve, to_t2 = _leave_out_curve(v, DROP_MAX)
    sv = pd.Series(v)
    rho = [float(sv.autocorr(k)) if len(v) > k + 2 else 0.0
           for k in range(1, RHO_LAGS + 1)]
    rho = [0.0 if not np.isfinite(r) else r for r in rho]
    g3 = float(sv.skew())
    g4 = float(sv.kurtosis()) + 3.0            # pandas gives EXCESS kurtosis

    opens = mark_open_close(blotter)
    return Weight(
        submission=submission,
        trades=len(blotter),
        round_trips=int(opens["is_open"].sum()),
        product_bets=len(product_pnl),
        delivery_days=n_active,
        calendar_days=len(v),
        products_per_day=mbar,
        icc=icc,
        design_effect=deff,
        rho1=float(pd.Series(v).autocorr(1)),
        eff_days=_participation_ratio(v),
        trades_per_day=len(blotter) / n_active if n_active else np.nan,
        sharpe=_annualised_sharpe(v),
        t_product=float(t_prod),
        t_day=float(t_day),
        ci_lo=float(np.percentile(sr_iid, 2.5)),
        ci_hi=float(np.percentile(sr_iid, 97.5)),
        blk_lo=float(np.percentile(sr_blk, 2.5)),
        blk_hi=float(np.percentile(sr_blk, 97.5)),
        drop_curve=curve,
        drop_to_t2=to_t2,
        skew=g3,
        kurtosis=g4,
        rho=rho,
        eta=_lo_eta(rho),
    )


def _verdict(w: Weight) -> str:
    """Weight of evidence, not quality of edge.

    A wide interval is not an accusation — it is a statement that this
    sample cannot support a confident verdict either way.
    """
    if not np.isfinite(w.ci_lo):
        return "INFO"
    if w.ci_lo <= 0.0:
        return "FAIL"          # the sample cannot separate the edge from zero
    if (w.eff_days < MIN_EFF_DAYS or w.ci_lo < SR_CI_FLOOR
            or w.design_effect >= DEFF_WARN
            or (w.drop_to_t2 is not None and w.drop_to_t2 < DROP_FRAGILE)):
        return "SUSPECT"
    return "PASS"


def evidence(registry_entry: dict, blotter: pd.DataFrame,
             product_pnl: pd.Series, daily: pd.Series) -> SectionResult:
    """The section result. Runs identically for every submission."""
    sub = str(registry_entry.get("submission", "?"))
    return evidence_from(measure(sub, blotter, product_pnl, daily))


def evidence_from(w: Weight) -> SectionResult:
    """Same result from an already-measured Weight.

    measure() runs 5,000 bootstrap resamples; callers that need both the
    Weight and the SectionResult should not pay for that twice.
    """
    v = _verdict(w)

    findings = [
        Finding("effective_sample_days", round(w.eff_days, 1),
                "SUSPECT" if w.eff_days < MIN_EFF_DAYS else "PASS",
                f"{w.trades:,} trades over {w.calendar_days} calendar days "
                f"reduce to {w.delivery_days} active delivery days and "
                f"{w.eff_days:.0f} effectively participating days "
                f"({w.trades_per_day:.0f} trades per active day)"),
        Finding("within_day_design_effect", round(w.design_effect, 2),
                "SUSPECT" if w.design_effect >= DEFF_WARN else "PASS",
                f"ICC={w.icc:+.2f} across {w.products_per_day:.1f} products "
                f"per day: counting every product bet as independent would "
                f"inflate t from {w.t_day:.1f} to {w.t_product:.1f} "
                f"({w.inflation:.1f}x)"),
        Finding("daily_autocorrelation", round(w.rho1, 3), "INFO",
                f"lag-1 rho={w.rho1:+.3f}; block bootstrap "
                f"[{w.blk_lo:.2f}, {w.blk_hi:.2f}] vs iid "
                f"[{w.ci_lo:.2f}, {w.ci_hi:.2f}] — days are serially "
                f"independent, so no block correction is needed"),
        Finding("days_to_insignificance", w.drop_to_t2,
                "SUSPECT" if (w.drop_to_t2 is not None
                              and w.drop_to_t2 < DROP_FRAGILE) else "PASS",
                (f"deleting the {w.drop_to_t2} best days of "
                 f"{w.calendar_days} pushes t below {T_FRAGILE:.0f}"
                 if w.drop_to_t2 is not None else
                 f"t stays above {T_FRAGILE:.0f} with the {DROP_MAX} best "
                 f"days removed — the edge is not carried by a few days")),
        Finding("sharpe_ci95", (round(w.ci_lo, 2), round(w.ci_hi, 2)),
                "FAIL" if w.ci_lo <= 0 else
                ("SUSPECT" if w.ci_lo < SR_CI_FLOOR else "PASS"),
                f"annualised Sharpe {w.sharpe:.2f}, bootstrapped at the "
                f"delivery-day unit over {w.calendar_days} days"),
    ]
    conds = []
    if v != "PASS":
        if w.eff_days < MIN_EFF_DAYS:
            conds.append(
                f"Its {w.trades:,} trades carry {w.eff_days:.0f} effective "
                f"observations, so treat the Sharpe with the uncertainty of a "
                f"{w.eff_days:.0f}-observation estimate however long the "
                f"{w.calendar_days}-day window looks. Re-validate on live days, "
                f"not on calendar time.")
        if w.design_effect >= DEFF_WARN:
            conds.append(
                f"Products inside one delivery day are worth "
                f"×{w.design_effect:.2f}, not {w.products_per_day:.0f} "
                f"independent bets (ICC {w.icc:+.2f}). Any later sizing or "
                f"significance work must count delivery days, not fills.")
        if w.drop_to_t2 is not None and w.drop_to_t2 < DROP_FRAGILE:
            conds.append(
                f"Deleting its {w.drop_to_t2} best days of {w.calendar_days} "
                f"pushes t below {T_FRAGILE:.0f}. Cap exposure so no single "
                f"session can carry the result, and set a kill on a drawdown "
                f"that removes that many good days.")
        if w.ci_lo < SR_CI_FLOOR:
            conds.append(
                f"The bootstrap lower bound on annualised Sharpe is "
                f"{w.ci_lo:.2f}. Paper-trade until the interval clears "
                f"{SR_CI_FLOOR:.1f} on live data.")
    return SectionResult("evidence", v, findings, conds)
