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
Checks 1-5 above are MANDATORY — all must pass for a stock to appear in the
report. Check 6 below is OPTIONAL / informational only as of this version:

  6. (optional) Current RSI back above its own fast RSI_SMA_PERIOD moving
     average (6a) AND the slower RSI_LONG_SMA_PERIOD ~1 month average (6b).
     This would confirm the reversal also holds up against the stock's own
     short/intermediate trend, not just against the raw trough number (e.g.
     DMart: RSI ~41 vs its own average ~55 — trough+bounce is real, but
     momentum hasn't turned on either timeframe yet). Stocks that pass 1-5
     but fail 6a/6b now still show up in the report — the "Trend" column
     (Uptrend / Downtrend / Neutral) tells you whether 6a/6b are holding, so
     you can see fresh-but-unconfirmed bounces (like HDFC Bank) instead of
     them being silently dropped.

Data source: Yahoo Finance via yfinance (tickers already carry the .NS suffix
in the watchlist below). No LLM. No external API key needed. Pure
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

# Watchlist — tickers already include the .NS suffix. Edit freely.
NIFTY_500_WATCHLIST = [
    # ── Rank 1–10 by Volume ──────────────────────────────────
    "ADANIPOWER.NS", "INFY.NS",       "WIPRO.NS",       "ETERNAL.NS",
    "JIOFIN.NS",     "HDFCBANK.NS",   "UNIONBANK.NS",   "TATASTEEL.NS",
    "KOTAKBANK.NS",  "VEDL.NS",
    # ── Rank 11–20 ───────────────────────────────────────────
    "CANBK.NS",      "ITC.NS",        "COALINDIA.NS",   "IRFC.NS",
    "ICICIBANK.NS",  "SBIN.NS",       "HINDZINC.NS",    "VBL.NS",
    "ADANIGREEN.NS", "ONGC.NS",
    # ── Rank 21–30 ───────────────────────────────────────────
    "RELIANCE.NS",   "BEL.NS",        "PNB.NS",         "MOTHERSON.NS",
    "HCLTECH.NS",    "BPCL.NS",       "POWERGRID.NS",   "SUNPHARMA.NS",
    "GAIL.NS",       "SHRIRAMFIN.NS",
    # ── Rank 31–40 ───────────────────────────────────────────
    "IOC.NS",        "PFC.NS",        "ADANIENSOL.NS",  "BANKBARODA.NS",
    "TATAPOWER.NS",  "BHARTIARTL.NS", "NTPC.NS",        "TATACAP.NS",
    "TMPV.NS",       "DRREDDY.NS",
    # ── Rank 41–50 ───────────────────────────────────────────
    "SBILIFE.NS",    "TCS.NS",        "RECLTD.NS",      "HINDALCO.NS",
    "TMCV.NS",       "CIPLA.NS",      "CGPOWER.NS",     "BAJFINANCE.NS",
    "GODREJCP.NS",   "AMBUJACEM.NS",
    # ── Rank 51–60 ───────────────────────────────────────────
    "TECHM.NS",      "AXISBANK.NS",   "NESTLEIND.NS",   "HDFCLIFE.NS",
    "MAXHEALTH.NS",  "M&M.NS",        "ADANIPORTS.NS",  "MAZDOCK.NS",
    "ADANIENT.NS",   "INDHOTEL.NS",
    # ── Rank 61–70 ───────────────────────────────────────────
    "LT.NS",         "DLF.NS",        "JSWSTEEL.NS",    "HINDUNILVR.NS",
    "TRENT.NS",      "LODHA.NS",      "TATACONSUM.NS",  "CHOLAFIN.NS",
    "JINDALSTEL.NS", "GRASIM.NS",
    # ── Rank 71–80 ───────────────────────────────────────────
    "HYUNDAI.NS",    "HDFCAMC.NS",    "UNITDSPR.NS",    "TITAN.NS",
    "LTM.NS",        "BAJAJFINSV.NS", "HAL.NS",         "TVSMOTOR.NS",
    "INDIGO.NS",     "ZYDUSLIFE.NS",
    # ── Rank 81–90 ───────────────────────────────────────────
    "MUTHOOTFIN.NS", "ENRIN.NS",      "PIDILITIND.NS",  "CUMMINSIND.NS",
    "BRITANNIA.NS",  "MARUTI.NS",     "ASIANPAINT.NS",  "EICHERMOT.NS",
    "APOLLOHOSP.NS", "ULTRACEMCO.NS",
    # ── Rank 91–100 ──────────────────────────────────────────
    "ABB.NS",        "DIVISLAB.NS",   "SIEMENS.NS",     "SOLARINDS.NS",
    "TORNTPHARM.NS", "DMART.NS",      "BAJAJ-AUTO.NS",  "BAJAJHLDNG.NS",
    "BOSCHLTD.NS",   "SHREECEM.NS",
    # ── Sector ETFs (BEES) ───────────────────────────────────
    "NIFTYBEES.NS",  "BANKBEES.NS",   "ITBEES.NS",      "AUTOBEES.NS",
    "PHARMABEES.NS", "GOLDBEES.NS",   "SILVERBEES.NS",
]


# Display-only metadata: maps ticker -> sector label, used purely for the
# HTML report's sector tags / heatmap panel. Does NOT affect scanning logic.
# Add/edit freely; unmapped tickers just show as "Other".
SECTOR_MAP = {
    "ADANIPOWER": "Power", "TATAPOWER": "Power", "POWERGRID": "Power",
    "NTPC": "Power", "ADANIENSOL": "Power", "ADANIGREEN": "Power",
    "INFY": "IT", "WIPRO": "IT", "TCS": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "ETERNAL": "Consumer", "VBL": "FMCG", "ITC": "FMCG", "GODREJCP": "FMCG",
    "NESTLEIND": "FMCG", "HINDUNILVR": "FMCG", "TATACONSUM": "FMCG",
    "BRITANNIA": "FMCG", "UNITDSPR": "FMCG",
    "JIOFIN": "Financials", "HDFCBANK": "Financials", "UNIONBANK": "Financials",
    "KOTAKBANK": "Financials", "CANBK": "Financials", "ICICIBANK": "Financials",
    "SBIN": "Financials", "PNB": "Financials", "SHRIRAMFIN": "Financials",
    "PFC": "Financials", "BANKBARODA": "Financials", "TATACAP": "Financials",
    "SBILIFE": "Financials", "RECLTD": "Financials", "BAJFINANCE": "Financials",
    "AXISBANK": "Financials", "HDFCLIFE": "Financials", "HDFCAMC": "Financials",
    "CHOLAFIN": "Financials", "BAJAJFINSV": "Financials", "MUTHOOTFIN": "Financials",
    "BAJAJHLDNG": "Financials",
    "TATASTEEL": "Metals", "VEDL": "Metals", "HINDZINC": "Metals",
    "HINDALCO": "Metals", "JINDALSTEL": "Metals", "COALINDIA": "Metals",
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", "GAIL": "Energy",
    "IOC": "Energy",
    "BEL": "Defence", "HAL": "Defence", "MAZDOCK": "Defence", "SOLARINDS": "Defence",
    "MOTHERSON": "Auto", "M&M": "Auto", "TVSMOTOR": "Auto", "MARUTI": "Auto",
    "EICHERMOT": "Auto", "BAJAJ-AUTO": "Auto", "BOSCHLTD": "Auto",
    "TMPV": "Auto", "TMCV": "Auto", "HYUNDAI": "Auto",
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "MAXHEALTH": "Pharma", "APOLLOHOSP": "Pharma", "DIVISLAB": "Pharma",
    "ZYDUSLIFE": "Pharma", "TORNTPHARM": "Pharma",
    "BHARTIARTL": "Telecom",
    "ADANIPORTS": "Infra", "DLF": "Realty", "LODHA": "Realty",
    "LT": "Infra", "GRASIM": "Cement", "AMBUJACEM": "Cement",
    "ULTRACEMCO": "Cement", "SHREECEM": "Cement",
    "ADANIENT": "Conglomerate", "INDHOTEL": "Hospitality", "TRENT": "Retail",
    "TITAN": "Consumer", "PIDILITIND": "Chemicals", "ASIANPAINT": "Chemicals",
    "CUMMINSIND": "Industrials", "CGPOWER": "Industrials", "ABB": "Industrials",
    "SIEMENS": "Industrials", "INDIGO": "Aviation", "DMART": "Retail",
    "IRFC": "Financials",
    "NIFTYBEES": "ETF", "BANKBEES": "ETF", "ITBEES": "ETF", "AUTOBEES": "ETF",
    "PHARMABEES": "ETF", "GOLDBEES": "ETF", "SILVERBEES": "ETF",
}


# Display-only: sector -> a small icon glyph for the header's sector-pulse
# ribbon. Purely cosmetic, does not affect scanning logic. Unmapped sectors
# just fall back to a plain dot.
SECTOR_ICONS = {
    "Power": "⚡", "IT": "💻", "FMCG": "🛒", "Consumer": "🛍",
    "Financials": "🏦", "Metals": "🪙", "Energy": "🛢",
    "Defence": "🛡", "Auto": "🚗", "Pharma": "💊", "Telecom": "📡",
    "Infra": "🏗", "Realty": "🏠", "Cement": "🧱", "Conglomerate": "🏢",
    "Hospitality": "🏨", "Retail": "🛍", "Chemicals": "🧪",
    "Industrials": "⚙", "Aviation": "✈", "ETF": "📊", "Other": "•",
}


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
    if checks is None:
        return None

    # Checks 1-5 are MANDATORY (real trough, real decline into it, real
    # recovery off it, no rollover since). Checks 6a/6b (RSI back above its
    # own fast/slow SMA) are informational only from here on -- a stock can
    # still signal without them, but the HTML "Trend" chip (Uptrend /
    # Downtrend / Neutral, driven by 6a+6b) shows whether momentum has
    # actually turned yet or the bounce is still happening under a falling
    # longer-term RSI trend.
    MANDATORY_KEYS = {"current_rsi", "trough_rsi", "decline_into_trough",
                       "recovery_points", "giveback"}
    mandatory_checks = [c for c in checks if c[4] in MANDATORY_KEYS]
    if not all(c[1] for c in mandatory_checks):
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
    total = len(NIFTY_500_WATCHLIST)

    for idx, ticker in enumerate(NIFTY_500_WATCHLIST, 1):
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


def _rsi_trend_chip(current_rsi, rsi_sma, rsi_long_sma):
    if current_rsi > rsi_sma and current_rsi > rsi_long_sma:
        return "Uptrend", "chip-up"
    elif current_rsi < rsi_sma and current_rsi < rsi_long_sma:
        return "Downtrend", "chip-down"
    return "Neutral", "chip-neutral"


def _render_panel_html(tf_key: str, results: list[dict], all_rsi: dict) -> tuple[str, int]:
    """Render one timeframe's sector-pulse grid + results table as a single
    <div class="tf-panel" data-tf="..."> block. Returns (panel_html, count).
    Kept as one self-contained chunk of markup so build_html can just loop
    over the three timeframes and paste each panel in, with only the
    active one visible on first paint (CSS/JS handles the rest)."""
    rows = ""
    signal_symbols = {r["symbol"] for r in results} if results else set()
    trend_counts = {"Uptrend": 0, "Downtrend": 0, "Neutral": 0}

    if results:
        for r in results:
            sector = SECTOR_MAP.get(r["symbol"], "Other")
            chg_class = "pos" if r["pct_chg"] >= 0 else "neg"
            rsi_color = _rsi_color(r["current_rsi"])
            trend_label, trend_class = _rsi_trend_chip(r["current_rsi"], r["rsi_sma"], r["rsi_long_sma"])
            trend_counts[trend_label] = trend_counts.get(trend_label, 0) + 1
            spark = _sparkline_svg([r["trough_rsi"], r["rsi_long_sma"], r["rsi_sma"], r["current_rsi"]])

            rows += f"""
            <tr data-symbol="{r['symbol']}" data-sector="{sector}" data-trend="{trend_label}"
                data-rsi="{r['current_rsi']}" data-recovery="{r['recovery_points']}"
                data-trough-age="{r['bars_since_trough']}" data-chg="{r['pct_chg']}">
                <td class="sym">{r['symbol']}<span class="sector-tag">{sector}</span></td>
                <td>{r['ltp']}</td>
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

    # Sector-pulse grid: EVERY sector that has at least one scanned stock on
    # this timeframe, sorted ascending (weakest RSI first -> strongest last).
    sector_rsi = {}
    sector_signals = {}
    for symbol, rsi_val in all_rsi.items():
        sector = SECTOR_MAP.get(symbol, "Other")
        sector_rsi.setdefault(sector, []).append(rsi_val)
        if symbol in signal_symbols:
            sector_signals[sector] = sector_signals.get(sector, 0) + 1

    ticker_html = ""
    if sector_rsi:
        sorted_sectors = sorted(
            sector_rsi.items(),
            key=lambda kv: (-sector_signals.get(kv[0], 0), sum(kv[1]) / len(kv[1]))
        )

        def _chip(sector, vals):
            avg = sum(vals) / len(vals)
            color = _rsi_color(avg)
            icon = SECTOR_ICONS.get(sector, "•")
            n_signals = sector_signals.get(sector, 0)
            signal_badge = (f'<span class="tick-signals">{n_signals} signal{"s" if n_signals != 1 else ""}</span>'
                             if n_signals else '<span class="tick-signals muted">0 signals</span>')
            return (f'<div class="tick-chip" data-sector="{sector}" tabindex="0" role="button" '
                    f'aria-pressed="false" style="border-color:{color}55">'
                    f'<div class="tick-row1">'
                    f'<span class="tick-icon">{icon}</span>'
                    f'<span class="tick-sector">{sector}</span>'
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
            <div class="ticker-label">Sector pulse &middot; most signals &rarr; fewest
                <span class="ticker-hint">&middot; click a sector to filter the table below</span>
            </div>
            <div class="ticker-grid">{chips}</div>
        </div>"""

    total_signals = len(results)
    trend_tabs_html = f"""
        <div class="trend-tabs" role="tablist" aria-label="Trend filter">
            <button class="trend-tab active" data-trend="all" role="tab" aria-selected="true">All <span class="trend-tab-count">{total_signals}</span></button>
            <button class="trend-tab" data-trend="Uptrend" role="tab" aria-selected="false">Uptrend <span class="trend-tab-count">{trend_counts['Uptrend']}</span></button>
            <button class="trend-tab" data-trend="Downtrend" role="tab" aria-selected="false">Downtrend <span class="trend-tab-count">{trend_counts['Downtrend']}</span></button>
            <button class="trend-tab" data-trend="Neutral" role="tab" aria-selected="false">Neutral <span class="trend-tab-count">{trend_counts['Neutral']}</span></button>
        </div>"""

    panel_html = f"""
    <div class="tf-panel" data-tf="{tf_key}">
        {ticker_html}
        {trend_tabs_html}
        <table>
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
                <tr class="no-sector-match" style="display:none">
                    <td colspan="11" class="empty">No signals match the current filter(s). Adjust or clear the sector/trend filter above.</td>
                </tr>
            </tbody>
        </table>
    </div>"""

    return panel_html, len(results)


def build_html(results_by_tf: dict[str, list], all_rsi_by_tf: dict[str, dict]) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tf_order = list(TIMEFRAMES.keys())  # ["1d", "1h", "15m"]

    panels_html = ""
    tab_buttons = ""
    tf_meta_js = "{"
    for i, tf_key in enumerate(tf_order):
        results = results_by_tf.get(tf_key, [])
        all_rsi = all_rsi_by_tf.get(tf_key, {})
        panel_html, count = _render_panel_html(tf_key, results, all_rsi)
        active = "active" if i == 0 else ""
        style = "" if i == 0 else ' style="display:none"'
        panels_html += panel_html.replace('class="tf-panel"', f'class="tf-panel {active}"', 1).replace(
            f'data-tf="{tf_key}">', f'data-tf="{tf_key}"{style}>', 1
        )
        tab_active = " active" if i == 0 else ""
        tab_buttons += (f'<button class="tf-tab{tab_active}" data-tf="{tf_key}" role="tab" '
                         f'aria-selected="{"true" if i == 0 else "false"}">{TIMEFRAMES[tf_key]["label"]}</button>')
        tf_meta_js += f'"{tf_key}":{{"label":"{TIMEFRAMES[tf_key]["label"]}","count":{count}}},'
    tf_meta_js = tf_meta_js.rstrip(",") + "}"

    first_tf = tf_order[0]
    first_count = len(results_by_tf.get(first_tf, []))
    first_label = TIMEFRAMES[first_tf]["label"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RSI Recovery Scan — All Timeframes (30-55 band, down &rarr; up)</title>
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

    .hdr-top {{
        display: flex; flex-direction: column; gap: 10px;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px; padding: 14px 18px 12px; margin-bottom: 10px;
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

    /* -- Tabs + toolbar card -- */
    .hdr-bottom {{
        display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 10px; padding: 10px 18px; margin-bottom: 14px;
    }}
    .tf-tabs {{ display: flex; gap: 6px; flex: 1 0 100%; margin-bottom: 4px; }}
    .tf-tab {{
        background: var(--panel2); color: var(--muted); border: 1px solid var(--border);
        border-radius: 999px; padding: 6px 16px; font-size: 12.5px; font-weight: 600;
        cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s;
    }}
    .tf-tab:hover {{ border-color: var(--accent); color: var(--text); }}
    .tf-tab.active {{ background: rgba(0,229,199,0.1); border-color: var(--accent); color: var(--accent); }}
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

    /* -- Per-timeframe panel: sector pulse + table -- */
    .tf-panel {{ display: flex; flex-direction: column; gap: 14px; }}

    .ticker-strip {{
        background: var(--panel2); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 18px 14px;
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

    /* -- Trend filter tabs (All / Uptrend / Downtrend / Neutral) -- */
    .trend-tabs {{
        display: flex; gap: 8px; flex-wrap: wrap;
    }}
    .trend-tab {{
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
        border-radius: 999px; padding: 6px 14px; font-size: 12px; font-weight: 600;
        cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s;
    }}
    .trend-tab:hover {{ border-color: var(--accent); color: var(--text); }}
    .trend-tab-count {{
        font-size: 10px; font-weight: 700; color: var(--muted);
        background: rgba(139,147,167,0.14); border-radius: 999px; padding: 1px 7px;
    }}
    .trend-tab.active {{ color: var(--text); border-color: var(--accent); background: rgba(0,229,199,0.08); }}
    .trend-tab.active .trend-tab-count {{ color: var(--accent); background: rgba(0,229,199,0.14); }}
    .trend-tab[data-trend="Uptrend"].active {{ border-color: var(--pos); background: rgba(38,208,124,0.10); }}
    .trend-tab[data-trend="Uptrend"].active .trend-tab-count {{ color: var(--pos); background: rgba(38,208,124,0.18); }}
    .trend-tab[data-trend="Downtrend"].active {{ border-color: var(--neg); background: rgba(255,84,112,0.10); }}
    .trend-tab[data-trend="Downtrend"].active .trend-tab-count {{ color: var(--neg); background: rgba(255,84,112,0.18); }}
    .trend-tab[data-trend="Neutral"].active {{ border-color: var(--muted); background: rgba(139,147,167,0.10); }}
    .spark {{ display: block; }}
    .empty {{ text-align: center; color: var(--muted); padding: 32px; }}
</style>
</head>
<body>
<div class="wrap">
    <div class="hdr-top">
        <div>
            <h1>RSI Recovery Scan — All Timeframes &middot; 30&ndash;55 band, down &rarr; up only</h1>
            <div class="sub" id="tfSubtitle">Generated {ts} &middot; <span class="count">{first_count}</span> stock(s) matched ({first_label})</div>
        </div>
        <div class="criteria-chips" title="RSI(14) formed a real trough within {TROUGH_WINDOW} candles (declining >= {PRIOR_DECLINE_MIN} pts over {PRIOR_DECLINE_BARS} bars), recovered >= {RECOVERY_MIN_POINTS} pts off that trough, sits within {WHIPSAW_TOLERANCE} pts of its post-trough high, and is back above both the {RSI_SMA_PERIOD}- and {RSI_LONG_SMA_PERIOD}-period RSI averages.">
            <span class="crit-chip tf">{first_label}</span>
            <span class="crit-chip band">RSI {RSI_BAND_LOW}&ndash;{RSI_BAND_HIGH}</span>
            <span class="crit-chip">Trough &le;<b>{TROUGH_WINDOW}</b>d</span>
            <span class="crit-chip">Decline &ge;<b>{PRIOR_DECLINE_MIN:g}</b>pt</span>
            <span class="crit-chip">Recovery &ge;<b>{RECOVERY_MIN_POINTS:g}</b>pt</span>
            <span class="crit-chip">Whipsaw &le;<b>{WHIPSAW_TOLERANCE:g}</b>pt</span>
            <span class="crit-chip">Above SMA<b>{RSI_SMA_PERIOD}</b>/<b>{RSI_LONG_SMA_PERIOD}</b></span>
        </div>
    </div>
    <div class="hdr-bottom">
        <div class="tf-tabs" role="tablist" aria-label="Timeframe">{tab_buttons}</div>
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
    {panels_html}
</div>
<script>
    const TF_ORDER = {tf_order!r};
    const TF_META = {tf_meta_js};
    let currentTF = TF_ORDER[0];
    const panelSort = {{}};
    const panelActiveSector = {{}};
    const panelActiveTrend = {{}};

    function panelEls(tf) {{
        const panel = document.querySelector('.tf-panel[data-tf="' + tf + '"]');
        const table = panel.querySelector('table');
        const tbody = table.querySelector('tbody');
        const sectorGrid = panel.querySelector('.ticker-grid');
        const noSectorMatchRow = panel.querySelector('.no-sector-match');
        return {{ panel, table, tbody, sectorGrid, noSectorMatchRow }};
    }}


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

    function dataRows(tf) {{
        const {{ tbody }} = panelEls(tf);
        return Array.from(tbody.querySelectorAll('tr')).filter(r => r.dataset.symbol);
    }}

    function sortRows(tf, key, type, dir) {{
        const {{ tbody }} = panelEls(tf);
        const rows = dataRows(tf);
        if (!rows.length) return;
        rows.sort((a, b) => {{
            const av = cellValue(a, key, type), bv = cellValue(b, key, type);
            if (av < bv) return -1 * dir;
            if (av > bv) return 1 * dir;
            return 0;
        }});
        rows.forEach(r => tbody.appendChild(r));
    }}

    // ---- Combined sector + trend filtering ----
    function applyFilters(tf) {{
        const {{ noSectorMatchRow }} = panelEls(tf);
        const sector = panelActiveSector[tf];
        const trend = panelActiveTrend[tf];
        const rows = dataRows(tf);
        let visible = 0;
        rows.forEach(r => {{
            const sectorOk = !sector || r.dataset.sector === sector;
            const trendOk = !trend || trend === 'all' || r.dataset.trend === trend;
            const match = sectorOk && trendOk;
            r.style.display = match ? '' : 'none';
            if (match) visible++;
        }});
        if (noSectorMatchRow) noSectorMatchRow.style.display = (visible === 0) ? '' : 'none';
    }}

    function setActiveSector(tf, sector, chipEl) {{
        panelActiveSector[tf] = sector;
        const {{ sectorGrid }} = panelEls(tf);
        sectorGrid.classList.toggle('filtering', !!sector);
        sectorGrid.querySelectorAll('.tick-chip').forEach(c => {{
            const isActive = sector && c === chipEl;
            c.classList.toggle('active', isActive);
            c.setAttribute('aria-pressed', isActive ? 'true' : 'false');
        }});
        applyFilters(tf);
    }}

    function setActiveTrend(tf, trend, btnEl) {{
        panelActiveTrend[tf] = trend === 'all' ? null : trend;
        const panel = document.querySelector('.tf-panel[data-tf="' + tf + '"]');
        const tabs = panel.querySelectorAll('.trend-tab');
        tabs.forEach(b => {{
            const isActive = b === btnEl;
            b.classList.toggle('active', isActive);
            b.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        applyFilters(tf);
    }}

    function initPanel(tf) {{
        const {{ table, sectorGrid, panel }} = panelEls(tf);
        panelSort[tf] = {{ key: null, dir: 1 }};
        panelActiveSector[tf] = null;
        panelActiveTrend[tf] = null;
        table.querySelectorAll('th[data-key]').forEach(th => {{
            th.addEventListener('click', () => {{
                const key = th.dataset.key, type = th.dataset.type;
                const dir = (panelSort[tf].key === key) ? -panelSort[tf].dir : -1;
                panelSort[tf] = {{ key, dir }};
                table.querySelectorAll('th').forEach(h => h.classList.remove('sorted'));
                th.classList.add('sorted');
                sortRows(tf, key, type, dir);
                const qf = document.getElementById('quickFilter');
                if (qf) qf.value = 'none';
            }});
        }});
        if (sectorGrid) {{
            sectorGrid.querySelectorAll('.tick-chip').forEach(chip => {{
                const sector = chip.dataset.sector;
                const toggle = () => {{
                    if (panelActiveSector[tf] === sector) setActiveSector(tf, null, null);
                    else setActiveSector(tf, sector, chip);
                }};
                chip.addEventListener('click', toggle);
                chip.addEventListener('keydown', e => {{
                    if (e.key === 'Enter' || e.key === ' ') {{ e.preventDefault(); toggle(); }}
                }});
            }});
        }}
        const trendTabs = panel.querySelectorAll('.trend-tab');
        trendTabs.forEach(btn => {{
            btn.addEventListener('click', () => setActiveTrend(tf, btn.dataset.trend, btn));
        }});
    }}
    TF_ORDER.forEach(initPanel);

    document.addEventListener('keydown', e => {{
        if (e.key === 'Escape' && panelActiveSector[currentTF]) setActiveSector(currentTF, null, null);
    }});

    // ---- Timeframe tabs ----
    function updateSubtitle(tf) {{
        const meta = TF_META[tf];
        document.getElementById('tfSubtitle').innerHTML =
            'Generated {ts} &middot; <span class="count">' + meta.count + '</span> stock(s) matched (' + meta.label + ')';
        const tfChip = document.querySelector('.crit-chip.tf');
        if (tfChip) tfChip.textContent = meta.label;
    }}

    function switchTF(tf) {{
        if (tf === currentTF) return;
        currentTF = tf;
        document.querySelectorAll('.tf-panel').forEach(p => {{
            const isActive = p.dataset.tf === tf;
            p.style.display = isActive ? '' : 'none';
            p.classList.toggle('active', isActive);
        }});
        document.querySelectorAll('.tf-tab').forEach(b => {{
            const isActive = b.dataset.tf === tf;
            b.classList.toggle('active', isActive);
            b.setAttribute('aria-selected', isActive ? 'true' : 'false');
        }});
        const qf = document.getElementById('quickFilter');
        if (qf) qf.value = 'none';
        updateSubtitle(tf);
    }}

    document.querySelectorAll('.tf-tab').forEach(btn => {{
        btn.addEventListener('click', () => switchTF(btn.dataset.tf));
    }});

    // ---- Export visible rows (active timeframe) to CSV ----
    function exportCSV() {{
        const {{ table, tbody }} = panelEls(currentTF);
        const headerCells = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
        const rows = Array.from(tbody.querySelectorAll('tr')).filter(r => r.dataset.symbol);
        if (!rows.length) return;
        const lines = [headerCells.join(',')];
        rows.forEach(r => {{
            const cells = Array.from(r.children).map(td => '"' + td.textContent.trim().replace(/"/g, '""') + '"');
            lines.push(cells.join(','));
        }});
        const blob = new Blob([lines.join('\\n')], {{ type: 'text/csv' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'rsi_recovery_scan_' + currentTF + '.csv';
        a.click();
        URL.revokeObjectURL(a.href);
    }}

    // ---- Alerts toggle (visual state only, global) ----
    let alertsOn = false;
    function toggleAlerts() {{
        alertsOn = !alertsOn;
        document.getElementById('alertsBtn').classList.toggle('active', alertsOn);
        document.getElementById('alertsState').textContent = alertsOn ? 'On' : 'Off';
    }}

    // ---- Quick filter dropdown (active timeframe) ----
    function applyQuickFilter(value) {{
        const {{ table }} = panelEls(currentTF);
        table.querySelectorAll('th').forEach(h => h.classList.remove('sorted'));
        if (value === 'rsi') {{ sortRows(currentTF, 'rsi', 'num', -1); panelSort[currentTF] = {{ key: 'rsi', dir: -1 }}; }}
        else if (value === 'recovery') {{ sortRows(currentTF, 'recovery', 'num', -1); panelSort[currentTF] = {{ key: 'recovery', dir: -1 }}; }}
        else if (value === 'trough') {{ sortRows(currentTF, 'trough-age', 'num', 1); panelSort[currentTF] = {{ key: 'trough-age', dir: 1 }}; }}
    }}
</script>
</body>
</html>"""


def debug_ticker(ticker: str, timeframe: str = DEFAULT_TIMEFRAME):
    """Print a full pass/fail breakdown of every check for a single ticker."""
    if not ticker.upper().endswith(".NS"):
        ticker = ticker.upper() + ".NS"
    tf = TIMEFRAMES[timeframe]
    print(f"Fetching {ticker} ({tf['label']}) ...")
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

    MANDATORY_KEYS = {"current_rsi", "trough_rsi", "decline_into_trough",
                       "recovery_points", "giveback"}
    all_mandatory_pass = True
    for name, passed, detail, _, key in checks:
        if key == "bars_since_trough":
            continue  # internal-only, already reflected in check 2's detail line
        status = "PASS" if passed else "FAIL"
        tag = "" if key in MANDATORY_KEYS else "  (optional -- informational only)"
        print(f"  [{status}] {name}{tag}\n         {detail}")
        if key in MANDATORY_KEYS and not passed:
            all_mandatory_pass = False

    print()
    if all_mandatory_pass:
        opt_failed = [name for name, passed, _, _, key in checks
                      if not passed and key not in MANDATORY_KEYS and key != "bars_since_trough"]
        if opt_failed:
            print(f"SIGNAL: all mandatory checks (1-5) passed. "
                  f"Optional 6a/6b not yet confirmed -- {', '.join(opt_failed)}. "
                  f"Will show in HTML with a Downtrend/Neutral trend chip.")
        else:
            print("SIGNAL: all checks (1-5 mandatory + 6a/6b optional) passed.")
    else:
        failed = [name for name, passed, _, _, key in checks
                  if not passed and key in MANDATORY_KEYS]
        print(f"NO SIGNAL: failed on mandatory check(s) -- {', '.join(failed)}")


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
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        if len(sys.argv) < 3:
            print("Usage: python3 rsi_recovery_scanner.py --debug TICKER [--timeframe 1d|1h|15m]")
            sys.exit(1)
        timeframe = _parse_timeframe(sys.argv)
        debug_ticker(sys.argv[2], timeframe=timeframe)
        return

    if "--timeframe" in sys.argv:
        print("Note: every run now scans all three timeframes (Daily, 1 Hour, 15 Min) "
              "into a single report with tabs to switch between them, so --timeframe is "
              "ignored here (it still works with --debug for a single-ticker check).\n")

    verbose = "--verbose" in sys.argv
    output_file = os.path.join(SCRIPT_DIR, "rsi_recovery_scanner.html")

    results_by_tf = {}
    all_rsi_by_tf = {}
    for tf_key, tf in TIMEFRAMES.items():
        print(f"Scanning {len(NIFTY_500_WATCHLIST)} stocks for RSI band recovery ({tf['label']}, down->up, 30-55)...")
        results, all_rsi = scan(verbose=verbose, timeframe=tf_key)
        results_by_tf[tf_key] = results
        all_rsi_by_tf[tf_key] = all_rsi
        print(f"  -> {len(results)} signal(s) found on {tf['label']}.\n")

    html = build_html(results_by_tf, all_rsi_by_tf)
    with open(output_file, "w") as f:
        f.write(html)

    summary = ", ".join(f"{TIMEFRAMES[k]['label']}: {len(v)}" for k, v in results_by_tf.items())
    print(f"Done. Report saved to {output_file} ({summary})")


if __name__ == "__main__":
    main()
