#!/usr/bin/env python3
"""
rsi_recovery_scanner.py

Deterministic scanner: finds stocks (from the watchlist below) where RSI(14)
is currently inside the 30-55 band AND trending UP (down -> up), never
stocks whose RSI is falling through that band from above (up -> down).

Signal criteria (all must hold):
  1. Current RSI is between RSI_BAND_LOW and RSI_BAND_HIGH (inclusive), i.e. 30-55.
  2. A real TROUGH (local low) in RSI is found within the last TROUGH_WINDOW
     candles, with at least one bar of room after it for a recovery leg.
  3. RSI genuinely DECLINED into that trough — fell by at least
     PRIOR_DECLINE_MIN points over the PRIOR_DECLINE_BARS candles before it.
     This is what rejects a stock whose RSI peaked (e.g. ~70) and has been
     sliding down for weeks/months with only a tiny 1-2 bar wobble at the very
     end — that is still an "up -> down" stock, not a genuine reversal.
  4. RSI has risen at least RECOVERY_MIN_POINTS off that trough. This is the
     actual "down -> up" test: a real, sizable move off the low (e.g. VEDL:
     25.03 -> 36.17, +11pts), not a 1-2 point twitch.
  5. No real rollover since the trough: today's RSI must sit within
     WHIPSAW_TOLERANCE points of the highest RSI reached since the trough.
     This absorbs ordinary single-day noise (VEDL's 36.27 -> 36.17, a 0.1pt
     dip) while still rejecting a genuine failed bounce / chop (Adani Power:
     spiked to ~53.73, then fell back to ~48.66 before today — a ~5pt
     rollover, well outside tolerance).
  6. Current RSI is back ABOVE both its own fast RSI_SMA_PERIOD moving
     average AND the slower RSI_LONG_SMA_PERIOD (~1 month) average. This
     rejects a stock that has a real trough+bounce but the bounce is a
     shallow one off a LOWER low within a still-intact downtrend (e.g. DMart:
     RSI ~41 vs its own average ~55) — the trough/decline/recovery checks
     alone can pass, but momentum hasn't actually turned yet on either
     timeframe.

Data source: Yahoo Finance via yfinance (plain US ticker symbols, no
exchange suffix needed). No LLM. No external API key needed. Pure
pandas/numpy RSI (Wilder's smoothing).

Output: self-contained dark-themed HTML report, written to the SAME FOLDER
as this script (not the current working directory), named
rsi_recovery_scanner.html

Usage:
    python3 rsi_recovery_scanner.py
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Install with: pip install yfinance --break-system-packages")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
RSI_BAND_LOW = 30          # current RSI must be >= this
RSI_BAND_HIGH = 55         # current RSI must be <= this
TROUGH_WINDOW = 20         # search this many recent bars for the RSI trough (local low)
PRIOR_DECLINE_BARS = 8     # bars BEFORE the trough that must show a real decline into it
PRIOR_DECLINE_MIN = 5.0    # RSI must have fallen by at least this many points into the trough
RECOVERY_MIN_POINTS = 5.0  # RSI must have risen at least this many points off the trough
WHIPSAW_TOLERANCE = 1.5    # allow today to sit up to this many points below the post-trough
                           # high without calling it a failed bounce (absorbs single-day noise
                           # like VEDL's 36.27 -> 36.17; a real rollover like Adani's
                           # 53.73 -> 48.66 is ~5 points and still gets rejected)
RSI_SMA_PERIOD = 9         # smoothing period for the RSI's own FAST moving average (signal line)
RSI_LONG_SMA_PERIOD = 21   # SLOWER, intermediate-term RSI average (~1 month on daily bars) — the real trend line

# Timeframe presets. All the checks above operate purely on the RSI series
# regardless of bar size, so switching timeframe only changes what yfinance
# interval/period is fetched — no scan/signal logic changes.
# NOTE: "20 bars", "8 bars" etc. above mean 20 DAYS on the daily timeframe,
# but only ~5 hours on 15m or ~3 trading days on 1h. The same thresholds are
# reused across timeframes for now; on intraday bars this generally makes the
# checks noisier/looser in real-world time, so expect more (and choppier)
# signals on 1h and especially 15m. Retune TROUGH_WINDOW / PRIOR_DECLINE_BARS
# / RSI_LONG_SMA_PERIOD per-timeframe later if that noise needs tightening.
TIMEFRAMES = {
    "1d":  {"interval": "1d",  "period": "6mo", "label": "Daily"},
    "1h":  {"interval": "1h",  "period": "60d", "label": "1 Hour"},
    "15m": {"interval": "15m", "period": "60d", "label": "15 Min"},
}
DEFAULT_TIMEFRAME = "1d"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _output_file(timeframe: str) -> str:
    suffix = "" if timeframe == "1d" else f"_{timeframe}"
    return os.path.join(SCRIPT_DIR, f"rsi_recovery_scanner{suffix}.html")

# Watchlist — S&P 500 large-caps. Plain yfinance symbols (no exchange
# suffix needed for US tickers). Edit freely.
#
# Display-only metadata: maps ticker -> sector ETF code (XLK, XLF, etc.),
# used purely for the HTML report's sector tags / sector-pulse row. Does NOT
# affect scanning logic. Unmapped tickers just show as "Other".
SECTOR_MAP = {
    **{s: "XLK" for s in [
        # Technology (16)
        "NVDA", "MSFT", "AAPL", "AVGO", "AMD", "ORCL", "ADBE", "PANW",
        "NOW", "SNPS", "CRM", "CSCO", "INTC", "QCOM", "AMAT", "LRCX",
        # Extras: SMCI
        "SMCI",
    ]},
    **{s: "XLC" for s in [
        # Communication Services (12)
        "GOOGL", "GOOG", "META", "NFLX", "CMCSA", "DIS",
        "TMUS", "VZ", "T", "CHTR", "SPOT", "RBLX",
    ]},
    **{s: "XLY" for s in [
        # Consumer Discretionary (13 — COST moved to Staples)
        "AMZN", "TSLA", "HD", "MCD", "TJX", "BKNG",
        "LOW", "SBUX", "NKE", "MAR", "ROST", "EBAY", "LULU",
    ]},
    **{s: "XLP" for s in [
        # Consumer Staples (10 — COST kept here as primary)
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL", "MNST",
    ]},
    **{s: "XLV" for s in [
        # Health Care (16)
        "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "AMGN", "BMY",
        "GILD", "ISRG", "VRTX", "CVS", "CI", "MDT", "SYK", "REGN",
    ]},
    **{s: "XLF" for s in [
        # Financials (16)
        "JPM", "BAC", "MS", "GS", "V", "MA", "AXP", "BLK",
        "SPGI", "C", "WFC", "SCHW", "COF", "PGR", "CB", "MMC",
        # Extras: HOOD, SOFI
        "HOOD", "SOFI",
    ]},
    **{s: "XLI" for s in [
        # Industrials (15)
        "GE", "CAT", "UNP", "HON", "LMT", "UPS", "RTX", "DE",
        "FDX", "BA", "GEV", "ETN", "ADP", "FAST", "CTAS",
    ]},
    **{s: "XLE" for s in [
        # Energy (12 — NEE, SO, DUK, CEG, VST kept here as listed)
        "XOM", "CVX", "COP", "NEE", "SO", "DUK", "CEG", "VST",
        "SLB", "EOG", "KMI", "PSX",
    ]},
    **{s: "XLB" for s in [
        # Materials (8)
        "LIN", "FCX", "SHW", "NEM", "APD", "ECL", "NUE", "DOW",
    ]},
    **{s: "XLRE" for s in [
        # Real Estate (10)
        "PLD", "AMT", "EQIX", "DLR", "WELL", "SPG", "PSA", "O", "CBRE", "VTR",
    ]},
    **{s: "XLU" for s in [
        # Utilities (10 — SO, DUK, NEE deduplicated to XLE above)
        "EXC", "XEL", "AEP", "SRE", "D", "PEG", "WEC", "ED", "EIX", "AWK",
    ]},
}

# Tickers to scan — derived automatically from SECTOR_MAP's keys.
US_WATCHLIST = list(SECTOR_MAP.keys())

# Display-only: sector ETF code -> (friendly label, small icon glyph) for the
# header's sector-pulse row. Purely cosmetic, does not affect scanning logic.
SECTOR_META = {
    "XLK":  ("Technology", "💻"),
    "XLC":  ("Comm Services", "📡"),
    "XLY":  ("Cons Discretionary", "🛍"),
    "XLP":  ("Cons Staples", "🛒"),
    "XLV":  ("Health Care", "💊"),
    "XLF":  ("Financials", "🏦"),
    "XLI":  ("Industrials", "⚙"),
    "XLE":  ("Energy", "🛢"),
    "XLB":  ("Materials", "🧪"),
    "XLRE": ("Real Estate", "🏠"),
    "XLU":  ("Utilities", "⚡"),
    "Other": ("Other", "•"),
}

MARKET_LABEL = "S&P 500 (US)"
CURRENCY = "$"


def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI (matches standard charting platforms)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)  # when avg_loss is 0, RSI = 100
    return rsi


def check_band_recovery(rsi: pd.Series, rsi_sma: pd.Series, rsi_long_sma: pd.Series) -> dict | None:
    """
    Return signal details only for a genuine down->up reversal — RSI that was
    truly falling, bottomed out, and has since turned up in a real way. NOT
    for stocks still drifting down with a tiny wobble, and NOT for stocks
    chopping sideways where an earlier bounce already failed.

    The shape being matched is exactly VEDL's: RSI slides for weeks, bottoms
    around 25, then rises 25.03 -> 33.60 -> 36.27 -> 36.17 — a big, real move
    off the low, with only a trivial 0.1pt dip on the last day. That trivial
    dip should NOT disqualify it; a stock like Adani Power that spiked ~5pts
    and then genuinely rolled over should.

    Checks:
      1. Current RSI sits inside the 30-55 band.
      2. A real TROUGH (local low) in RSI is found within the last
         TROUGH_WINDOW candles, with at least one bar of room after it.
      3. RSI genuinely DECLINED into that trough beforehand — at least
         PRIOR_DECLINE_MIN points over PRIOR_DECLINE_BARS candles. Rejects a
         stock whose RSI peaked and has been sliding for weeks/months with
         only a tiny 1-2 bar wobble at the very end (e.g. Asian Paints).
      4. RSI has risen at least RECOVERY_MIN_POINTS off that trough. This is
         the "down -> up" test itself — a real, sizable move off the low, not
         a 1-2 point twitch that happens to sit above the low.
      5. No real whipsaw since the trough: today's RSI must be within
         WHIPSAW_TOLERANCE points of the highest RSI reached since the
         trough. This absorbs ordinary single-day noise (VEDL's 36.27 ->
         36.17) while still rejecting a genuine failed bounce (Adani:
         ~53.73 -> ~48.66, a ~5pt rollover, well outside tolerance).
      6. Current RSI is back above both its own fast (RSI_SMA_PERIOD) and
         slow (RSI_LONG_SMA_PERIOD) moving averages — confirms the reversal
         also holds up against the stock's own short and intermediate trend,
         not just against the raw trough number.
    """
    min_len = TROUGH_WINDOW + PRIOR_DECLINE_BARS + RSI_LONG_SMA_PERIOD + 1
    checks = diagnose_checks(rsi, rsi_sma, rsi_long_sma, min_len)
    if checks is None or not all(c[1] for c in checks):
        return None

    vals = {c[4]: c[3] for c in checks}
    return {
        "current_rsi": round(vals["current_rsi"], 2),
        "trough_rsi": round(vals["trough_rsi"], 2),
        "bars_since_trough": vals["bars_since_trough"],
        "recovery_points": round(vals["recovery_points"], 2),
        "rsi_sma": round(vals["current_sma"], 2),
        "rsi_long_sma": round(vals["current_long_sma"], 2),
    }


def diagnose_checks(rsi: pd.Series, rsi_sma: pd.Series, rsi_long_sma: pd.Series,
                     min_len: int) -> list | None:
    """
    Walks through every check UNCONDITIONALLY (no early-exit) and returns a
    list of (name, passed: bool, detail: str, raw_value) tuples so a caller
    can both (a) decide pass/fail overall, and (b) print exactly which
    check(s) failed and why. Returns None only if there isn't even enough
    data to compute the checks at all.
    """
    if len(rsi) < min_len:
        return None

    current_rsi = float(rsi.iloc[-1])
    current_sma = float(rsi_sma.iloc[-1])
    current_long_sma = float(rsi_long_sma.iloc[-1])
    out = []

    # 1. Must be inside the band right now
    p = RSI_BAND_LOW <= current_rsi <= RSI_BAND_HIGH
    out.append((f"1. In band [{RSI_BAND_LOW}-{RSI_BAND_HIGH}]", p,
                 f"current RSI = {current_rsi:.2f}", current_rsi, "current_rsi"))

    # 2. Find the trough: the lowest RSI value within the last TROUGH_WINDOW bars
    recent_window = rsi.iloc[-TROUGH_WINDOW:]
    trough_pos_in_window = int(recent_window.values.argmin())
    trough_idx = len(rsi) - TROUGH_WINDOW + trough_pos_in_window
    trough_rsi = float(rsi.iloc[trough_idx])
    bars_since_trough = (len(rsi) - 1) - trough_idx
    p = bars_since_trough >= 1
    out.append(("2. Trough exists with room for a recovery leg (>= 1 bar ago)", p,
                 f"trough_rsi = {trough_rsi:.2f}, {bars_since_trough} bar(s) ago", trough_rsi, "trough_rsi"))
    out.append(("2b. (internal) bars since trough", p,
                 f"{bars_since_trough} bar(s) ago", bars_since_trough, "bars_since_trough"))

    # 3. Confirm RSI genuinely DECLINED into the trough beforehand
    prior_start_idx = max(0, trough_idx - PRIOR_DECLINE_BARS)
    prior_start_rsi = float(rsi.iloc[prior_start_idx])
    decline_into_trough = prior_start_rsi - trough_rsi
    p = decline_into_trough >= PRIOR_DECLINE_MIN
    out.append((f"3. Genuine decline into trough (>= {PRIOR_DECLINE_MIN} pts over {PRIOR_DECLINE_BARS} bars)", p,
                 f"declined {decline_into_trough:.2f} pts ({prior_start_rsi:.2f} -> {trough_rsi:.2f})", decline_into_trough, "decline_into_trough"))

    # 4. The actual "down -> up" test: RSI must have risen a REAL amount off
    #    the trough, not just be a point or two above it.
    recovery_points = current_rsi - trough_rsi
    p = recovery_points >= RECOVERY_MIN_POINTS
    out.append((f"4. Recovered >= {RECOVERY_MIN_POINTS} pts off trough", p,
                 f"recovered {recovery_points:.2f} pts ({trough_rsi:.2f} -> {current_rsi:.2f})", recovery_points, "recovery_points"))

    # 5. No real whipsaw since the trough: today can sit a little below the
    #    post-trough high (ordinary noise) but not meaningfully below it
    #    (a genuine failed bounce / rollover).
    since_trough = rsi.iloc[trough_idx + 1: -1]  # bars after trough, before today
    since_trough_max = float(since_trough.max()) if not since_trough.empty else current_rsi
    giveback = since_trough_max - current_rsi
    p = giveback <= WHIPSAW_TOLERANCE
    out.append((f"5. No real rollover since trough (giveback <= {WHIPSAW_TOLERANCE} pts)", p,
                 f"giveback {giveback:.2f} pts (post-trough high {since_trough_max:.2f} -> today {current_rsi:.2f})", giveback, "giveback"))

    # 6. Confirm the reversal holds up against both the fast and slow RSI averages
    p = current_rsi > current_sma
    out.append((f"6a. RSI above fast SMA{RSI_SMA_PERIOD}", p,
                 f"current {current_rsi:.2f} vs SMA{RSI_SMA_PERIOD} {current_sma:.2f}", current_sma, "current_sma"))

    p = current_rsi > current_long_sma
    out.append((f"6b. RSI above slow SMA{RSI_LONG_SMA_PERIOD}", p,
                 f"current {current_rsi:.2f} vs SMA{RSI_LONG_SMA_PERIOD} {current_long_sma:.2f}", current_long_sma, "current_long_sma"))

    return out


def scan(verbose: bool = False, timeframe: str = DEFAULT_TIMEFRAME) -> tuple[list[dict], dict]:
    """
    Scan the whole watchlist on the given timeframe ("1d", "1h", or "15m").
    By default this stays quiet — a single overwriting progress line — and
    only the final summary is printed by main(). Pass verbose=True (or run
    with --verbose) to get the old per-ticker [idx/total] SYMBOL ...
    SIGNAL/no signal/error/skip lines.

    Returns (results, all_rsi):
      - results: list of dicts for tickers that fired the full band-recovery
        signal (as before).
      - all_rsi: {ticker: current_rsi} for EVERY ticker that had enough
        history to compute RSI, whether or not it signalled. This feeds the
        sector-pulse row so every sector is represented, not just sectors
        that happen to have a signal today.
    """
    tf = TIMEFRAMES[timeframe]
    results = []
    all_rsi = {}
    total = len(US_WATCHLIST)

    for idx, ticker in enumerate(US_WATCHLIST, 1):
        if verbose:
            print(f"[{idx}/{total}] {ticker} ...", end=" ", flush=True)
        else:
            print(f"\rScanning ({tf['label']})... [{idx}/{total}] {ticker:<16}", end="", flush=True)
        try:
            df = yf.Ticker(ticker).history(period=tf["period"], interval=tf["interval"])
            # Yahoo sometimes returns a still-forming/incomplete final candle
            # with a NaN Close (e.g. mid-session, or a brief data hiccup).
            # Drop trailing NaN-Close rows so "last row" always means the
            # last real, closed candle — otherwise LTP/Chg% come out NaN
            # while RSI silently reuses the prior day's value (ewm() carries
            # the last valid average forward through a NaN input instead of
            # producing NaN itself), which is worse: it looks fine but isn't.
            df = df[df["Close"].notna()]
            min_needed = (RSI_PERIOD + TROUGH_WINDOW + PRIOR_DECLINE_BARS
                          + RSI_LONG_SMA_PERIOD)
            if df.empty or len(df) < min_needed:
                if verbose:
                    print("skip (insufficient data)")
                continue

            rsi = compute_rsi(df["Close"])
            rsi_sma = rsi.rolling(RSI_SMA_PERIOD).mean()
            rsi_long_sma = rsi.rolling(RSI_LONG_SMA_PERIOD).mean()
            all_rsi[ticker.replace(".NS", "")] = round(float(rsi.iloc[-1]), 2)
            signal = check_band_recovery(rsi, rsi_sma, rsi_long_sma)

            if signal:
                last_close = round(float(df["Close"].iloc[-1]), 2)
                prev_close = round(float(df["Close"].iloc[-2]), 2)
                pct_chg = round((last_close - prev_close) / prev_close * 100, 2)
                results.append({
                    "symbol": ticker.replace(".NS", ""),
                    "ltp": last_close,
                    "pct_chg": pct_chg,
                    **signal,
                })
                if verbose:
                    print(f"SIGNAL (RSI {signal['current_rsi']}, trough {signal['trough_rsi']}, sma9 {signal['rsi_sma']}, sma21 {signal['rsi_long_sma']})")
            elif verbose:
                print("no signal")
        except Exception as e:
            if verbose:
                print(f"error: {e}")
            continue

    if not verbose:
        print("\r" + " " * 60 + "\r", end="", flush=True)  # clear the progress line

    results.sort(key=lambda r: r["current_rsi"])
    return results, all_rsi


def _rsi_color(v: float) -> str:
    """Red -> yellow -> green gradient for an RSI value (0-100)."""
    v = max(0.0, min(100.0, v))
    if v <= 50:
        t = v / 50.0
        r1, g1, b1 = 0xff, 0x54, 0x70   # red (--neg)
        r2, g2, b2 = 0xf5, 0xc5, 0x18   # yellow
    else:
        t = (v - 50) / 50.0
        r1, g1, b1 = 0xf5, 0xc5, 0x18   # yellow
        r2, g2, b2 = 0x26, 0xd0, 0x7c   # green (--pos)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"rgb({r},{g},{b})"


def _sparkline_svg(points: list) -> str:
    """Tiny inline SVG polyline sparkline for trough -> long_sma -> sma -> now."""
    w, h, pad = 72, 24, 3
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    n = len(points)
    coords = []
    for i, p in enumerate(points):
        x = pad + (i / (n - 1)) * (w - 2 * pad)
        y = h - pad - ((p - lo) / span) * (h - 2 * pad)
        coords.append(f"{x:.1f},{y:.1f}")
    last_x, last_y = coords[-1].split(",")
    poly = " ".join(coords)
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<polyline points="{poly}" fill="none" stroke="var(--accent)" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="{last_x}" cy="{last_y}" r="2" fill="var(--accent)"/></svg>')


def build_html(results: list[dict], all_rsi: dict | None = None, timeframe: str = DEFAULT_TIMEFRAME) -> str:
    tf_label = TIMEFRAMES[timeframe]["label"]
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = ""
    all_rsi = all_rsi or {}

    # Which symbols fired a signal today, for the per-sector signal-count badge.
    signal_symbols = {r["symbol"] for r in results} if results else set()

    if results:
        for r in results:
            sector_code = SECTOR_MAP.get(r["symbol"], "Other")
            sector_label, _ = SECTOR_META.get(sector_code, (sector_code, "•"))

            chg_class = "pos" if r["pct_chg"] >= 0 else "neg"
            rsi_color = _rsi_color(r["current_rsi"])

            # Trend chip: signals already require RSI above both SMAs, so this
            # will read "Uptrend" for every row today, but stays correct if
            # the thresholds/criteria are ever loosened.
            if r["current_rsi"] > r["rsi_sma"] and r["current_rsi"] > r["rsi_long_sma"]:
                trend_label, trend_class = "Uptrend", "chip-up"
            elif r["current_rsi"] < r["rsi_sma"] and r["current_rsi"] < r["rsi_long_sma"]:
                trend_label, trend_class = "Downtrend", "chip-down"
            else:
                trend_label, trend_class = "Neutral", "chip-neutral"

            spark = _sparkline_svg([r["trough_rsi"], r["rsi_long_sma"], r["rsi_sma"], r["current_rsi"]])

            rows += f"""
            <tr data-symbol="{r['symbol']}" data-sector="{sector_label}"
                data-rsi="{r['current_rsi']}" data-recovery="{r['recovery_points']}"
                data-trough-age="{r['bars_since_trough']}" data-chg="{r['pct_chg']}">
                <td class="sym">{r['symbol']}<span class="sector-tag">{sector_label}</span></td>
                <td>{CURRENCY}{r['ltp']}</td>
                <td class="{chg_class}">{r['pct_chg']:+.2f}%</td>
                <td><span class="rsi-badge" style="background:{rsi_color}22;color:{rsi_color};border:1px solid {rsi_color}55">{r['current_rsi']}</span></td>
                <td>{r['trough_rsi']}</td>
                <td>{r['bars_since_trough']}d ago</td>
                <td>{r['rsi_sma']}</td>
                <td>{r['rsi_long_sma']}</td>
                <td>{spark}</td>
                <td><span class="chip {trend_class}">{trend_label}</span></td>
                <td>+{r['recovery_points']} pts</td>
            </tr>"""
    else:
        rows = '<tr><td colspan="11" class="empty">No stocks currently match the RSI band-recovery criteria.</td></tr>'

    # Sector-pulse grid: EVERY sector that has at least one scanned stock,
    # not just sectors with a signal today. Average RSI is computed across
    # all successfully-scanned stocks in that sector, so a sector shows up
    # even if it currently has zero band-recovery signals. Sorted ascending
    # (weakest/most-oversold sector first -> strongest last), i.e. "down to
    # up", matching the down->up recovery logic the whole scanner is built
    # around. Rendered as a full-width grid so it fills the row instead of
    # leaving space unused.
    sector_rsi = {}       # sector code -> list of current_rsi across ALL scanned stocks
    sector_signals = {}   # sector code -> count of stocks with a live signal today
    for symbol, rsi_val in all_rsi.items():
        sector_code = SECTOR_MAP.get(symbol, "Other")
        sector_rsi.setdefault(sector_code, []).append(rsi_val)
        if symbol in signal_symbols:
            sector_signals[sector_code] = sector_signals.get(sector_code, 0) + 1

    ticker_html = ""
    if sector_rsi:
        sorted_sectors = sorted(sector_rsi.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))

        def _chip(sector_code, vals):
            avg = sum(vals) / len(vals)
            color = _rsi_color(avg)
            label, icon = SECTOR_META.get(sector_code, (sector_code, "•"))
            n_signals = sector_signals.get(sector_code, 0)
            signal_badge = (f'<span class="tick-signals">{n_signals} signal{"s" if n_signals != 1 else ""}</span>'
                             if n_signals else '<span class="tick-signals muted">0 signals</span>')
            return (f'<div class="tick-chip" data-sector="{sector_label}" tabindex="0" role="button" '
                    f'aria-pressed="false" style="border-color:{color}55">'
                    f'<div class="tick-row1">'
                    f'<span class="tick-icon">{icon}</span>'
                    f'<span class="tick-sector">{label}</span>'
                    f'</div>'
                    f'<div class="tick-row2">'
                    f'<span class="tick-rsi" style="color:{color}">{avg:.1f}</span>'
                    f'<span class="tick-count">{len(vals)} stocks</span>'
                    f'</div>'
                    f'<div class="tick-bar"><span style="width:{avg:.1f}%;background:{color}"></span></div>'
                    f'{signal_badge}'
                    f'</div>')

        chips = "".join(_chip(s, v) for s, v in sorted_sectors)
        ticker_html = f"""
    <div class="ticker-strip" role="group" aria-label="Sector pulse">
        <div class="ticker-label">Sector pulse &middot; weakest &rarr; strongest RSI
            <span class="ticker-hint" id="sectorFilterHint">&middot; click a sector to filter the table below</span>
        </div>
        <div class="ticker-grid" id="sectorGrid">{chips}</div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSI Recovery Scan — {MARKET_LABEL} · {tf_label} (30-55 band, down &rarr; up)</title>
<style>
    :root {{
        --bg: #0b0e14; --panel: #131722; --panel2: #171c29; --border: #232838;
        --text: #e6e9f0; --muted: #8b93a7; --accent: #00e5c7;
        --pos: #26d07c; --neg: #ff5470;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
        margin: 0; padding: 32px;
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    h1 {{ font-size: 21px; margin: 0 0 3px; font-weight: 700; letter-spacing: -0.01em; }}
    .sub {{ color: var(--muted); font-size: 12.5px; font-weight: 300; }}
    .count {{ color: var(--accent); font-weight: 700; }}

    /* ============ Layered header: top / mid / bottom strips ============ */
    .hdr {{ margin-bottom: 18px; }}

    /* -- Top strip: title + criteria summary chips -- */
    .hdr-top {{
        display: flex; flex-direction: column; gap: 10px;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px 10px 0 0; padding: 14px 18px 12px;
        border-bottom: none;
    }}
    .criteria-chips {{
        display: flex; flex-wrap: nowrap; gap: 6px;
        overflow-x: auto; padding-bottom: 2px; scrollbar-width: thin;
    }}
    .criteria-chips::-webkit-scrollbar {{ height: 4px; }}
    .criteria-chips::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
    .crit-chip {{
        font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
        font-size: 10.5px; font-weight: 600; letter-spacing: 0.02em;
        padding: 4px 10px; border-radius: 999px; white-space: nowrap; flex: 0 0 auto;
        background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
    }}
    .crit-chip b {{ color: var(--text); font-weight: 700; }}
    .crit-chip.band {{
        color: #0b0e14; font-weight: 700; border: none;
        background: linear-gradient(90deg, var(--neg) 0%, #f5c518 50%, var(--pos) 100%);
    }}
    .crit-chip.tf {{ color: var(--accent); border-color: rgba(0,229,199,0.35); background: rgba(0,229,199,0.08); }}
    .crit-chip.market {{ color: var(--text); font-weight: 700; border-color: var(--border); background: var(--panel); }}

    /* -- Middle strip: sector-pulse grid, ALL sectors, weakest->strongest RSI -- */
    .ticker-strip {{
        display: block;
        background: var(--panel2); border: 1px solid var(--border); border-top: none;
        padding: 12px 18px 14px;
    }}
    .ticker-label {{
        font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: var(--muted); margin-bottom: 10px;
    }}
    .ticker-hint {{
        text-transform: none; font-weight: 500; letter-spacing: normal;
        color: var(--muted); opacity: 0.75; font-size: 10px;
    }}
    .ticker-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 8px;
    }}
    .tick-chip {{
        display: flex; flex-direction: column; gap: 4px;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 8px; padding: 8px 10px;
        cursor: pointer; user-select: none;
        transition: opacity 0.15s, border-color 0.15s, box-shadow 0.15s, transform 0.1s;
    }}
    .tick-chip:hover {{ border-color: var(--accent); }}
    .tick-chip:active {{ transform: scale(0.98); }}
    .tick-chip.active {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent);
        background: rgba(0,229,199,0.08);
    }}
    .ticker-grid.filtering .tick-chip:not(.active) {{ opacity: 0.4; }}
    .tick-row1 {{ display: flex; align-items: center; gap: 6px; }}
    .tick-row2 {{ display: flex; align-items: baseline; justify-content: space-between; gap: 6px; }}
    .tick-icon {{ font-size: 12px; }}
    .tick-sector {{ font-size: 11px; font-weight: 500; color: var(--muted);
        overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .tick-rsi {{ font-size: 16px; font-weight: 700; font-stretch: condensed; letter-spacing: -0.02em; }}
    .tick-count {{ font-size: 9.5px; color: var(--muted); background: var(--panel2);
        border-radius: 999px; padding: 1px 6px; white-space: nowrap; }}
    .tick-bar {{ height: 3px; border-radius: 999px; background: var(--panel2); overflow: hidden; }}
    .tick-bar span {{ display: block; height: 100%; border-radius: 999px; }}
    .tick-signals {{ font-size: 9.5px; font-weight: 600; color: var(--pos); }}
    .tick-signals.muted {{ color: var(--muted); font-weight: 500; }}

    /* -- Bottom strip: quick filters + status/actions bar -- */
    .hdr-bottom {{
        display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
        background: var(--panel); border: 1px solid var(--border); border-top: none;
        border-radius: 0 0 10px 10px; padding: 10px 18px;
    }}
    .toolbar-left {{ display: flex; align-items: center; gap: 10px; }}
    .toolbar-left label {{ font-size: 12px; color: var(--muted); }}
    .toolbar-left select {{
        background: var(--panel2); color: var(--text); border: 1px solid var(--border);
        border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer;
    }}
    .toolbar-right {{ display: flex; align-items: center; gap: 8px; }}
    .action-btn {{
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--panel2); color: var(--text); border: 1px solid var(--border);
        border-radius: 8px; padding: 6px 12px; font-size: 12.5px; font-weight: 600;
        cursor: pointer; transition: border-color 0.15s, color 0.15s;
    }}
    .action-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
    .action-btn.active {{ border-color: var(--accent); color: var(--accent); background: rgba(0,229,199,0.08); }}
    .user-pill {{
        display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted);
        background: var(--panel2); border: 1px solid var(--border); border-radius: 999px; padding: 5px 12px 5px 6px;
    }}
    .user-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--pos); box-shadow: 0 0 0 3px rgba(38,208,124,0.18); }}

    table {{ width: 100%; border-collapse: collapse; background: var(--panel);
        border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
    th {{ text-align: left; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--muted); padding: 10px 14px;
        border-bottom: 1px solid var(--border); cursor: pointer; user-select: none;
        white-space: nowrap; }}
    th:hover {{ color: var(--text); }}
    th.sorted::after {{ content: " \\2195"; color: var(--accent); }}
    td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 13px; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .sym {{ font-weight: 600; }}
    .sector-tag {{ display: inline-block; margin-left: 8px; font-size: 10px; font-weight: 500;
        color: var(--muted); background: var(--panel2); border: 1px solid var(--border);
        border-radius: 5px; padding: 1px 6px; text-transform: uppercase; letter-spacing: 0.03em; }}
    .pos {{ color: var(--pos); }}
    .neg {{ color: var(--neg); }}
    .rsi-badge {{ font-weight: 700; padding: 2px 9px; border-radius: 999px; font-size: 12.5px; }}
    .chip {{ padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.03em; }}
    .chip-up {{ background: rgba(38,208,124,0.14); color: var(--pos); }}
    .chip-down {{ background: rgba(255,84,112,0.14); color: var(--neg); }}
    .chip-neutral {{ background: rgba(139,147,167,0.14); color: var(--muted); }}
    .spark {{ display: block; }}
    .empty {{ text-align: center; color: var(--muted); padding: 32px; }}
</style>
</head>
<body>
<div class="wrap">
    <div class="hdr">
        <div class="hdr-top">
            <div>
                <h1>RSI Recovery Scan — {MARKET_LABEL} &middot; {tf_label} &middot; 30&ndash;55 band, down &rarr; up only</h1>
                <div class="sub">Generated {ts} · <span class="count">{len(results)}</span> stock(s) matched</div>
            </div>
            <div class="criteria-chips" title="RSI(14) formed a real trough within {TROUGH_WINDOW} candles (declining >= {PRIOR_DECLINE_MIN} pts over {PRIOR_DECLINE_BARS} bars), recovered >= {RECOVERY_MIN_POINTS} pts off that trough, sits within {WHIPSAW_TOLERANCE} pts of its post-trough high, and is back above both the {RSI_SMA_PERIOD}- and {RSI_LONG_SMA_PERIOD}-period RSI averages.">
                <span class="crit-chip market">{MARKET_LABEL}</span>
                <span class="crit-chip tf">{tf_label}</span>
                <span class="crit-chip band">RSI {RSI_BAND_LOW}&ndash;{RSI_BAND_HIGH}</span>
                <span class="crit-chip">Trough &le;<b>{TROUGH_WINDOW}</b>d</span>
                <span class="crit-chip">Decline &ge;<b>{PRIOR_DECLINE_MIN:g}</b>pt</span>
                <span class="crit-chip">Recovery &ge;<b>{RECOVERY_MIN_POINTS:g}</b>pt</span>
                <span class="crit-chip">Whipsaw &le;<b>{WHIPSAW_TOLERANCE:g}</b>pt</span>
                <span class="crit-chip">Above SMA<b>{RSI_SMA_PERIOD}</b>/<b>{RSI_LONG_SMA_PERIOD}</b></span>
            </div>
        </div>
        {ticker_html}
        <div class="hdr-bottom">
            <div class="toolbar-left">
                <label for="quickFilter">Quick filter</label>
                <select id="quickFilter" onchange="applyQuickFilter(this.value)">
                    <option value="none">All (default order)</option>
                    <option value="rsi">Strongest RSI</option>
                    <option value="recovery">Top Recovery</option>
                    <option value="trough">Recent Troughs</option>
                </select>
            </div>
            <div class="toolbar-right">
                <button class="action-btn" id="exportBtn" onclick="exportCSV()">&#8681; Export</button>
                <button class="action-btn" id="alertsBtn" onclick="toggleAlerts()">&#128276; Alerts: <span id="alertsState">Off</span></button>
                <span class="user-pill"><span class="user-dot"></span>Scanner &middot; Live</span>
            </div>
        </div>
    </div>
    <table id="resultsTable">
        <thead>
            <tr>
                <th data-key="symbol" data-type="text">Symbol</th>
                <th data-key="ltp" data-type="num">LTP</th>
                <th data-key="chg" data-type="num">Chg %</th>
                <th data-key="rsi" data-type="num">RSI (now)</th>
                <th data-key="trough" data-type="num">RSI (trough)</th>
                <th data-key="trough-age" data-type="num">Trough</th>
                <th data-key="sma" data-type="num">SMA{RSI_SMA_PERIOD}</th>
                <th data-key="longsma" data-type="num">SMA{RSI_LONG_SMA_PERIOD}</th>
                <th>Recovery shape</th>
                <th>Trend</th>
                <th data-key="recovery" data-type="num">Recovery</th>
            </tr>
        </thead>
        <tbody>{rows}
            <tr id="noSectorMatchRow" class="no-sector-match" style="display:none">
                <td colspan="11" class="empty">No signal today in this sector. Click the sector again to clear the filter.</td>
            </tr>
        </tbody>
    </table>
</div>
<script>
    // ---- Interactive column sorting ----
    const table = document.getElementById('resultsTable');
    const tbody = table.querySelector('tbody');
    let sortState = {{ key: null, dir: 1 }};

    function cellValue(row, key, type) {{
        const map = {{
            symbol: () => row.dataset.symbol,
            ltp: () => parseFloat(row.children[1].textContent),
            chg: () => parseFloat(row.dataset.chg),
            rsi: () => parseFloat(row.dataset.rsi),
            trough: () => parseFloat(row.children[4].textContent),
            'trough-age': () => parseFloat(row.dataset.troughAge),
            sma: () => parseFloat(row.children[6].textContent),
            longsma: () => parseFloat(row.children[7].textContent),
            recovery: () => parseFloat(row.dataset.recovery),
        }};
        const v = map[key] ? map[key]() : '';
        return type === 'text' ? String(v) : (isNaN(v) ? -Infinity : v);
    }}

    function sortRows(key, type, dir) {{
        const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => r.dataset.symbol);
        if (!rows.length) return;
        rows.sort((a, b) => {{
            const av = cellValue(a, key, type), bv = cellValue(b, key, type);
            if (av < bv) return -1 * dir;
            if (av > bv) return 1 * dir;
            return 0;
        }});
        rows.forEach(r => tbody.appendChild(r));
    }}

    table.querySelectorAll('th[data-key]').forEach(th => {{
        th.addEventListener('click', () => {{
            const key = th.dataset.key, type = th.dataset.type;
            const dir = (sortState.key === key) ? -sortState.dir : -1;
            sortState = {{ key, dir }};
            table.querySelectorAll('th').forEach(h => h.classList.remove('sorted'));
            th.classList.add('sorted');
            sortRows(key, type, dir);
            document.getElementById('quickFilter').value = 'none';
        }});
    }});

    // ---- Export visible rows to CSV ----
    function exportCSV() {{
        const headerCells = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
        const dataRows = Array.from(tbody.querySelectorAll('tr')).filter(r => r.dataset.symbol);
        if (!dataRows.length) return;
        const lines = [headerCells.join(',')];
        dataRows.forEach(r => {{
            const cells = Array.from(r.children).map(td => '"' + td.textContent.trim().replace(/"/g, '""') + '"');
            lines.push(cells.join(','));
        }});
        const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'rsi_recovery_scan.csv';
        a.click();
        URL.revokeObjectURL(a.href);
    }}

    // ---- Alerts toggle (visual state only) ----
    let alertsOn = false;
    function toggleAlerts() {{
        alertsOn = !alertsOn;
        document.getElementById('alertsBtn').classList.toggle('active', alertsOn);
        document.getElementById('alertsState').textContent = alertsOn ? 'On' : 'Off';
    }}

    // ---- Quick filter dropdown ----
    function applyQuickFilter(value) {{
        table.querySelectorAll('th').forEach(h => h.classList.remove('sorted'));
        if (value === 'rsi') {{ sortRows('rsi', 'num', -1); sortState = {{ key: 'rsi', dir: -1 }}; }}
        else if (value === 'recovery') {{ sortRows('recovery', 'num', -1); sortState = {{ key: 'recovery', dir: -1 }}; }}
        else if (value === 'trough') {{ sortRows('trough-age', 'num', 1); sortState = {{ key: 'trough-age', dir: 1 }}; }}
    }}

    // ---- Sector-pulse click-to-filter ----
    // Click a sector card to show only that sector's signal rows in the
    // table below. Click the same card again (or press Escape) to clear
    // the filter and show every row.
    let activeSector = null;
    const sectorGrid = document.getElementById('sectorGrid');
    const noSectorMatchRow = document.getElementById('noSectorMatchRow');

    function dataRows() {{
        return Array.from(tbody.querySelectorAll('tr')).filter(r => r.dataset.symbol);
    }}

    function applySectorFilter(sector) {{
        const rows = dataRows();
        let visible = 0;
        rows.forEach(r => {{
            const match = r.dataset.sector === sector;
            r.style.display = match ? '' : 'none';
            if (match) visible++;
        }});
        if (noSectorMatchRow) noSectorMatchRow.style.display = (visible === 0) ? '' : 'none';
    }}

    function clearSectorFilter() {{
        dataRows().forEach(r => {{ r.style.display = ''; }});
        if (noSectorMatchRow) noSectorMatchRow.style.display = 'none';
    }}

    function setActiveSector(sector, chipEl) {{
        activeSector = sector;
        sectorGrid.classList.toggle('filtering', !!sector);
        sectorGrid.querySelectorAll('.tick-chip').forEach(c => {{
            const isActive = sector && c === chipEl;
            c.classList.toggle('active', isActive);
            c.setAttribute('aria-pressed', isActive ? 'true' : 'false');
        }});
        if (sector) applySectorFilter(sector);
        else clearSectorFilter();
    }}

    if (sectorGrid) {{
        sectorGrid.querySelectorAll('.tick-chip').forEach(chip => {{
            const sector = chip.dataset.sector;
            const toggle = () => {{
                if (activeSector === sector) setActiveSector(null, null);
                else setActiveSector(sector, chip);
            }};
            chip.addEventListener('click', toggle);
            chip.addEventListener('keydown', e => {{
                if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); toggle(); }}
            }});
        }});
        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape' && activeSector) setActiveSector(null, null);
        }});
    }}
</script>
</body>
</html>"""


def debug_ticker(ticker: str, timeframe: str = DEFAULT_TIMEFRAME):
    """Print a full pass/fail breakdown of every check for a single ticker."""
    ticker = ticker.upper()
    tf = TIMEFRAMES[timeframe]
    print(f"Fetching {ticker} ({MARKET_LABEL}, {tf['label']}) ...")
    df = yf.Ticker(ticker).history(period=tf["period"], interval=tf["interval"])
    if df.empty:
        print("No data returned for this ticker.")
        return
    df = df[df["Close"].notna()]  # drop a still-forming/incomplete final candle
    if df.empty:
        print("No usable (closed) candles returned for this ticker.")
        return

    rsi = compute_rsi(df["Close"])
    rsi_sma = rsi.rolling(RSI_SMA_PERIOD).mean()
    rsi_long_sma = rsi.rolling(RSI_LONG_SMA_PERIOD).mean()
    min_len = TROUGH_WINDOW + PRIOR_DECLINE_BARS + RSI_LONG_SMA_PERIOD + 1

    checks = diagnose_checks(rsi, rsi_sma, rsi_long_sma, min_len)
    print(f"\n{ticker} — last close {float(df['Close'].iloc[-1]):.2f}, "
          f"as of {df.index[-1].date()}\n")

    if checks is None:
        print(f"Not enough history: need >= {min_len} bars, have {len(rsi)}.")
        return

    all_pass = True
    for name, passed, detail, _, key in checks:
        if key == "bars_since_trough":
            continue  # internal-only, already reflected in check 2's detail line
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}\n         {detail}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("SIGNAL: all checks passed.")
    else:
        failed = [name for name, passed, _, _, key in checks
                  if not passed and key != "bars_since_trough"]
        print(f"NO SIGNAL: failed on {len(failed)} check(s) — {', '.join(failed)}")


def _parse_timeframe(argv: list) -> str:
    if "--timeframe" in argv:
        i = argv.index("--timeframe")
        if i + 1 >= len(argv):
            print(f"Usage: --timeframe {{{'|'.join(TIMEFRAMES)}}}")
            sys.exit(1)
        tf = argv[i + 1]
        if tf not in TIMEFRAMES:
            print(f"Unknown timeframe '{tf}'. Choose from: {', '.join(TIMEFRAMES)}")
            sys.exit(1)
        return tf
    return DEFAULT_TIMEFRAME


def main():
    timeframe = _parse_timeframe(sys.argv)

    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        if len(sys.argv) < 3:
            print("Usage: python3 rsi_recovery_scanner.py --debug TICKER [--timeframe 1d|1h|15m]")
            sys.exit(1)
        debug_ticker(sys.argv[2], timeframe=timeframe)
        return

    verbose = "--verbose" in sys.argv
    output_file = _output_file(timeframe)
    tf_label = TIMEFRAMES[timeframe]["label"]

    print(f"Scanning {len(US_WATCHLIST)} stocks for RSI band recovery ({MARKET_LABEL}, {tf_label}, down->up, 30-55)...\n")
    results, all_rsi = scan(verbose=verbose, timeframe=timeframe)
    html = build_html(results, all_rsi, timeframe=timeframe)
    with open(output_file, "w") as f:
        f.write(html)
    print(f"Done. {len(results)} signal(s) found. Report saved to {output_file}")


if __name__ == "__main__":
    main()
