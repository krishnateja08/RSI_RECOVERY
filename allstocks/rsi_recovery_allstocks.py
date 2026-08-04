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

Data source: Yahoo Finance via yfinance. No LLM. No external API key needed.
Pure pandas/numpy RSI (Wilder's smoothing).

Watchlist: by default this scans the FULL live NIFTY 200 (India) and S&P 500
(USA) constituent lists — fetched fresh on every run from NSE's archives and
Wikipedia respectively, no manual ticker maintenance needed. See section
"LIVE WATCHLIST" below for the fallback/toggle behavior
(USE_FULL_NIFTY200 / USE_FULL_SP500 env vars).

NOTE ON RUNTIME: scanning ~700 tickers across all 3 timeframes used to mean
~2,100 sequential Yahoo Finance requests (one ticker at a time, each with a
0.2s sleep) -- 30-90+ minutes and a real chance of tripping rate limiting.
As of this version, Daily and 15-Min requests for every ticker are queued
into ONE shared thread pool (see fetch_multi_timeframe_parallel /
MAX_FETCH_WORKERS in main()), so a single stock's Daily and 15-Min fetches
can run at the same time rather than in two back-to-back all-tickers
passes. 1-Hour bars are derived from the already-fetched 15-Min data by
resampling instead of a third fetch -- so it's ~2x710 overlapping requests
instead of 3x710 serial ones. If it's still too slow/flaky, set
USE_FULL_NIFTY200=false and/or USE_FULL_SP500=false to fall back to the
much smaller curated lists further down this file, or lower
MAX_FETCH_WORKERS if you hit rate limits.

Output: self-contained dark-themed HTML report, written to the SAME FOLDER
as this script (not the current working directory), named
rsi_recovery_scanner.html

Usage:
    python3 rsi_recovery_scanner.py
"""

import os
import sys
import csv
import io
import time
import datetime
from zoneinfo import ZoneInfo
import urllib.request
import urllib.error
import concurrent.futures
import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Install with: pip install yfinance pandas lxml --break-system-packages")
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

# ---------------------------------------------------------------------------
# FIBONACCI CONFIG
# ---------------------------------------------------------------------------
# Applied ONLY to stocks that already passed the RSI recovery signal above —
# purely informational context ("if I'm buying this, where's the Fib range
# and what's a sane entry/stop/target"). It never changes which stocks show
# up in the report; that's still decided 100% by the RSI checks. Computed
# off the SAME OHLC data already fetched for the RSI scan (no extra
# network calls), using the same recent-swing zigzag logic as
# fib_swing_trade_plan.py.
FIB_ZIGZAG_PCT = 5.0       # zigzag reversal threshold (%) to confirm a swing pivot — same value/rationale
                           # as fib_swing_trade_plan.py; reused across all 3 timeframes for now (see the
                           # TIMEFRAMES note above re: thresholds not yet retuned per-timeframe)
FIB_SWING_ORDER = 8        # fallback local-extrema window for very short/quiet price histories
FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSION = 1.618
FIB_MIN_BARS = 30          # skip Fib calc below this many bars — not enough history for a meaningful swing

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

# Toggle whether to scan the FULL live NIFTY 200 / S&P 500 constituent
# lists (recommended) or fall back to the small curated lists below.
# Override via env vars, e.g. for a quick test run or if the live sources
# are ever unreachable from your network:
#   USE_FULL_NIFTY200=false
#   USE_FULL_SP500=false
USE_FULL_NIFTY200 = os.environ.get("USE_FULL_NIFTY200", "true").strip().lower() in ("1", "true", "yes", "on")
USE_FULL_SP500 = os.environ.get("USE_FULL_SP500", "true").strip().lower() in ("1", "true", "yes", "on")

# Watchlist — this ~100-company curated list (tickers already carry the .NS
# suffix) is now only a FALLBACK, used per-market if the corresponding live
# fetch below fails or is disabled. Edit freely.
INDIA_WATCHLIST_STATIC = [
    # Full live NIFTY 200 constituent list (fallback copy, captured Aug 2026 -
    # kept in sync manually; the live fetch above supersedes this whenever it
    # succeeds). Plus the 7 sector ETFs (BEES) appended at the end.
    "360ONE.NS", "ABB.NS", "APLAPOLLO.NS", "AUBANK.NS", "ADANIENSOL.NS",
    "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS", "ATGL.NS",
    "ABCAPITAL.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASHOKLEY.NS",
    "ASIANPAINT.NS", "ASTRAL.NS", "AUROPHARMA.NS", "DMART.NS", "AXISBANK.NS",
    "BSE.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS",
    "BANKBARODA.NS", "BANKINDIA.NS", "BDL.NS", "BEL.NS", "BHARATFORG.NS",
    "BHEL.NS", "BPCL.NS", "BHARTIARTL.NS", "GROWW.NS", "BIOCON.NS",
    "BLUESTARCO.NS", "BOSCHLTD.NS", "BRITANNIA.NS", "CGPOWER.NS", "CANBK.NS",
    "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COCHINSHIP.NS", "COFORGE.NS",
    "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CUMMINSIND.NS", "DLF.NS",
    "DABUR.NS", "DIVISLAB.NS", "DIXON.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "ETERNAL.NS", "EXIDEIND.NS", "NYKAA.NS", "FEDERALBNK.NS", "FORTIS.NS",
    "GAIL.NS", "GVT&D.NS", "GMRAIRPORT.NS", "GLENMARK.NS", "GODFRYPHLP.NS",
    "GODREJCP.NS", "GODREJPROP.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCAMC.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HAVELLS.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HAL.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "POWERINDIA.NS",
    "HUDCO.NS", "HYUNDAI.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIAMC.NS",
    "IDFCFIRSTB.NS", "ITC.NS", "INDIANB.NS", "INDHOTEL.NS", "IOC.NS",
    "IRCTC.NS", "IRFC.NS", "IREDA.NS", "INDUSTOWER.NS", "INDUSINDBK.NS",
    "NAUKRI.NS", "INFY.NS", "INDIGO.NS", "JSWENERGY.NS", "JSWSTEEL.NS",
    "JINDALSTEL.NS", "JIOFIN.NS", "JUBLFOOD.NS", "KEI.NS", "KPITTECH.NS",
    "KALYANKJIL.NS", "KOTAKBANK.NS", "LTF.NS", "LGEINDIA.NS", "LICHSGFIN.NS",
    "LTM.NS", "LT.NS", "LAURUSLABS.NS", "LENSKART.NS", "LODHA.NS",
    "LUPIN.NS", "MRF.NS", "M&MFIN.NS", "M&M.NS", "MANKIND.NS",
    "MARICO.NS", "MARUTI.NS", "MFSL.NS", "MAXHEALTH.NS", "MAZDOCK.NS",
    "MOTILALOFS.NS", "MPHASIS.NS", "MCX.NS", "MUTHOOTFIN.NS", "NHPC.NS",
    "NMDC.NS", "NTPC.NS", "NATIONALUM.NS", "NESTLEIND.NS", "OBEROIRLTY.NS",
    "ONGC.NS", "OIL.NS", "PAYTM.NS", "OFSS.NS", "POLICYBZR.NS",
    "PIIND.NS", "PAGEIND.NS", "PATANJALI.NS", "PERSISTENT.NS", "PHOENIXLTD.NS",
    "PIDILITIND.NS", "POLYCAB.NS", "PFC.NS", "POWERGRID.NS", "PREMIERENE.NS",
    "PRESTIGE.NS", "PNB.NS", "RECLTD.NS", "RADICO.NS", "RVNL.NS",
    "RELIANCE.NS", "SBICARD.NS", "SBILIFE.NS", "SRF.NS", "MOTHERSON.NS",
    "SHREECEM.NS", "SHRIRAMFIN.NS", "ENRIN.NS", "SIEMENS.NS", "SOLARINDS.NS",
    "SBIN.NS", "SAIL.NS", "SUNPHARMA.NS", "SUPREMEIND.NS", "SUZLON.NS",
    "SWIGGY.NS", "TVSMOTOR.NS", "TATACAP.NS", "TATACOMM.NS", "TCS.NS",
    "TATACONSUM.NS", "TATAELXSI.NS", "TATAINVEST.NS", "TMCV.NS", "TMPV.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS",
    "TRENT.NS", "TIINDIA.NS", "UPL.NS", "ULTRACEMCO.NS", "UNIONBANK.NS",
    "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "VMM.NS", "IDEA.NS",
    "VOLTAS.NS", "WAAREEENER.NS", "WIPRO.NS", "YESBANK.NS", "ZYDUSLIFE.NS",
    # ── Sector ETFs (BEES) ───────────────────────────────────
    "NIFTYBEES.NS", "BANKBEES.NS", "ITBEES.NS", "AUTOBEES.NS",
    "PHARMABEES.NS", "GOLDBEES.NS", "SILVERBEES.NS",
]

# USA fallback — only used if the live S&P 500 fetch fails or is disabled
# (USE_FULL_SP500=false). A small set of large caps across sectors so the
# report still has USA coverage even without a live fetch. Tickers carry NO
# suffix (bare Yahoo Finance format, e.g. "AAPL"), same as the live list.
USA_WATCHLIST_STATIC = [
    # Full live S&P 500 constituent list (fallback copy, captured Aug 2026 -
    # kept in sync manually; the live fetch above supersedes this whenever it
    # succeeds). Tickers use bare Yahoo Finance format (e.g. "BRK-B").
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES",
    "AFL", "A", "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN",
    "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR",
    "AEE", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "AME",
    "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ",
    "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON",
    "BKR", "BALL", "BAC", "BAX", "BDX", "BRK-B", "BBY", "TECH",
    "BIIB", "BLK", "BX", "XYZ", "BNY", "BA", "BKNG", "BSX",
    "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP",
    "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR",
    "CVNA", "CASY", "CAT", "CBOE", "CBRE", "CDW", "COR", "CNC",
    "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB",
    "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO", "C", "CFG",
    "CLX", "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL",
    "CMCSA", "FIX", "CAG", "COP", "ED", "STZ", "CEG", "COO",
    "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CRH", "CRWD",
    "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA",
    "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR",
    "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI",
    "DTE", "DUK", "DD", "ETN", "EBAY", "SATS", "ECL", "EIX",
    "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG", "EPAM",
    "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG",
    "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM",
    "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
    "FSLR", "FE", "FISV", "F", "FTNT", "FTV", "FOXA", "FOX",
    "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN",
    "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL",
    "GDDY", "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC",
    "HSY", "HPE", "HLT", "HD", "HON", "HRL", "HST", "HWM",
    "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX",
    "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF",
    "IP", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT",
    "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP",
    "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC",
    "KR", "LHX", "LH", "LRCX", "LVS", "LDOS", "LEN", "LII",
    "LLY", "LIN", "LYV", "LMT", "L", "LOW", "LULU", "LITE",
    "LYB", "MTB", "MPC", "MAR", "MRSH", "MLM", "MAS", "MA",
    "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD",
    "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "TAP", "MDLZ",
    "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ",
    "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI",
    "NDSN", "NSC", "NTRS", "NOC", "NCLH", "NRG", "NUE", "NVDA",
    "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", "ON", "OKE",
    "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH",
    "PAYX", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX",
    "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR",
    "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM",
    "DGX", "Q", "RL", "RJF", "RTX", "O", "REG", "REGN",
    "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL", "ROP",
    "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX",
    "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA",
    "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE",
    "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO",
    "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN",
    "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT",
    "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER",
    "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", "UHS",
    "VLO", "VEEV", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX",
    "VRT", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
    "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC",
    "WELL", "WST", "WDC", "WY", "WSM", "WMB", "WTW", "WDAY",
    "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
]

# NSE's own "Industry" classification (from the live NIFTY 200 constituent
# file) is used directly as the sector label for any ticker pulled in via
# the live fetch below — no separate mapping table needed, so newly-added
# constituents are labeled automatically. Icons for these labels (plus the
# older hand-picked labels used by INDIA_WATCHLIST_STATIC/USA_WATCHLIST_STATIC
# above and the GICS labels used for live S&P 500 tickers) are all defined
# in SECTOR_ICONS below.


# Display-only metadata: maps ticker -> sector label, used purely for the
# HTML report's sector tags / heatmap panel. Does NOT affect scanning logic.
# Add/edit freely; unmapped tickers just show as "Other".
SECTOR_MAP = {
    # Auto-added first: full NIFTY 200 (NSE Industry) + full S&P 500 (GICS
    # Sector) labels, captured Aug 2026, so every ticker in the full static
    # fallback lists above has a sector for the report's heatmap/tags.
    # Hand-picked entries further below override these where more specific.
    "360ONE": "Financial Services",
    "ABB": "Capital Goods",
    "APLAPOLLO": "Capital Goods",
    "AUBANK": "Financial Services",
    "ADANIENSOL": "Power",
    "ADANIENT": "Metals & Mining",
    "ADANIGREEN": "Power",
    "ADANIPORTS": "Services",
    "ADANIPOWER": "Power",
    "ATGL": "Oil Gas & Consumable Fuels",
    "ABCAPITAL": "Financial Services",
    "ALKEM": "Healthcare",
    "AMBUJACEM": "Construction Materials",
    "APOLLOHOSP": "Healthcare",
    "ASHOKLEY": "Capital Goods",
    "ASIANPAINT": "Consumer Durables",
    "ASTRAL": "Capital Goods",
    "AUROPHARMA": "Healthcare",
    "DMART": "Consumer Services",
    "AXISBANK": "Financial Services",
    "BSE": "Financial Services",
    "BAJAJ-AUTO": "Automobile and Auto Components",
    "BAJFINANCE": "Financial Services",
    "BAJAJFINSV": "Financial Services",
    "BAJAJHLDNG": "Financial Services",
    "BANKBARODA": "Financial Services",
    "BANKINDIA": "Financial Services",
    "BDL": "Capital Goods",
    "BEL": "Capital Goods",
    "BHARATFORG": "Automobile and Auto Components",
    "BHEL": "Capital Goods",
    "BPCL": "Oil Gas & Consumable Fuels",
    "BHARTIARTL": "Telecommunication",
    "GROWW": "Financial Services",
    "BIOCON": "Healthcare",
    "BLUESTARCO": "Consumer Durables",
    "BOSCHLTD": "Automobile and Auto Components",
    "BRITANNIA": "Fast Moving Consumer Goods",
    "CGPOWER": "Capital Goods",
    "CANBK": "Financial Services",
    "CHOLAFIN": "Financial Services",
    "CIPLA": "Healthcare",
    "COALINDIA": "Oil Gas & Consumable Fuels",
    "COCHINSHIP": "Capital Goods",
    "COFORGE": "Information Technology",
    "COLPAL": "Fast Moving Consumer Goods",
    "CONCOR": "Services",
    "COROMANDEL": "Chemicals",
    "CUMMINSIND": "Capital Goods",
    "DLF": "Realty",
    "DABUR": "Fast Moving Consumer Goods",
    "DIVISLAB": "Healthcare",
    "DIXON": "Consumer Durables",
    "DRREDDY": "Healthcare",
    "EICHERMOT": "Automobile and Auto Components",
    "ETERNAL": "Consumer Services",
    "EXIDEIND": "Automobile and Auto Components",
    "NYKAA": "Consumer Services",
    "FEDERALBNK": "Financial Services",
    "FORTIS": "Healthcare",
    "GAIL": "Oil Gas & Consumable Fuels",
    "GVT&D": "Capital Goods",
    "GMRAIRPORT": "Services",
    "GLENMARK": "Healthcare",
    "GODFRYPHLP": "Fast Moving Consumer Goods",
    "GODREJCP": "Fast Moving Consumer Goods",
    "GODREJPROP": "Realty",
    "GRASIM": "Construction Materials",
    "HCLTECH": "Information Technology",
    "HDFCAMC": "Financial Services",
    "HDFCBANK": "Financial Services",
    "HDFCLIFE": "Financial Services",
    "HAVELLS": "Consumer Durables",
    "HEROMOTOCO": "Automobile and Auto Components",
    "HINDALCO": "Metals & Mining",
    "HAL": "Capital Goods",
    "HINDPETRO": "Oil Gas & Consumable Fuels",
    "HINDUNILVR": "Fast Moving Consumer Goods",
    "HINDZINC": "Metals & Mining",
    "POWERINDIA": "Capital Goods",
    "HUDCO": "Financial Services",
    "HYUNDAI": "Automobile and Auto Components",
    "ICICIBANK": "Financial Services",
    "ICICIGI": "Financial Services",
    "ICICIAMC": "Financial Services",
    "IDFCFIRSTB": "Financial Services",
    "ITC": "Fast Moving Consumer Goods",
    "INDIANB": "Financial Services",
    "INDHOTEL": "Consumer Services",
    "IOC": "Oil Gas & Consumable Fuels",
    "IRCTC": "Consumer Services",
    "IRFC": "Financial Services",
    "IREDA": "Financial Services",
    "INDUSTOWER": "Telecommunication",
    "INDUSINDBK": "Financial Services",
    "NAUKRI": "Consumer Services",
    "INFY": "Information Technology",
    "INDIGO": "Services",
    "JSWENERGY": "Power",
    "JSWSTEEL": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining",
    "JIOFIN": "Financial Services",
    "JUBLFOOD": "Consumer Services",
    "KEI": "Capital Goods",
    "KPITTECH": "Information Technology",
    "KALYANKJIL": "Consumer Durables",
    "KOTAKBANK": "Financial Services",
    "LTF": "Financial Services",
    "LGEINDIA": "Consumer Durables",
    "LICHSGFIN": "Financial Services",
    "LTM": "Information Technology",
    "LT": "Construction",
    "LAURUSLABS": "Healthcare",
    "LENSKART": "Consumer Services",
    "LODHA": "Realty",
    "LUPIN": "Healthcare",
    "MRF": "Automobile and Auto Components",
    "M&MFIN": "Financial Services",
    "M&M": "Automobile and Auto Components",
    "MANKIND": "Healthcare",
    "MARICO": "Fast Moving Consumer Goods",
    "MARUTI": "Automobile and Auto Components",
    "MFSL": "Financial Services",
    "MAXHEALTH": "Healthcare",
    "MAZDOCK": "Capital Goods",
    "MOTILALOFS": "Financial Services",
    "MPHASIS": "Information Technology",
    "MCX": "Financial Services",
    "MUTHOOTFIN": "Financial Services",
    "NHPC": "Power",
    "NMDC": "Metals & Mining",
    "NTPC": "Power",
    "NATIONALUM": "Metals & Mining",
    "NESTLEIND": "Fast Moving Consumer Goods",
    "OBEROIRLTY": "Realty",
    "ONGC": "Oil Gas & Consumable Fuels",
    "OIL": "Oil Gas & Consumable Fuels",
    "PAYTM": "Financial Services",
    "OFSS": "Information Technology",
    "POLICYBZR": "Financial Services",
    "PIIND": "Chemicals",
    "PAGEIND": "Textiles",
    "PATANJALI": "Fast Moving Consumer Goods",
    "PERSISTENT": "Information Technology",
    "PHOENIXLTD": "Realty",
    "PIDILITIND": "Chemicals",
    "POLYCAB": "Capital Goods",
    "PFC": "Financial Services",
    "POWERGRID": "Power",
    "PREMIERENE": "Capital Goods",
    "PRESTIGE": "Realty",
    "PNB": "Financial Services",
    "RECLTD": "Financial Services",
    "RADICO": "Fast Moving Consumer Goods",
    "RVNL": "Construction",
    "RELIANCE": "Oil Gas & Consumable Fuels",
    "SBICARD": "Financial Services",
    "SBILIFE": "Financial Services",
    "SRF": "Chemicals",
    "MOTHERSON": "Automobile and Auto Components",
    "SHREECEM": "Construction Materials",
    "SHRIRAMFIN": "Financial Services",
    "ENRIN": "Capital Goods",
    "SIEMENS": "Capital Goods",
    "SOLARINDS": "Chemicals",
    "SBIN": "Financial Services",
    "SAIL": "Metals & Mining",
    "SUNPHARMA": "Healthcare",
    "SUPREMEIND": "Capital Goods",
    "SUZLON": "Capital Goods",
    "SWIGGY": "Consumer Services",
    "TVSMOTOR": "Automobile and Auto Components",
    "TATACAP": "Financial Services",
    "TATACOMM": "Telecommunication",
    "TCS": "Information Technology",
    "TATACONSUM": "Fast Moving Consumer Goods",
    "TATAELXSI": "Information Technology",
    "TATAINVEST": "Financial Services",
    "TMCV": "Capital Goods",
    "TMPV": "Automobile and Auto Components",
    "TATAPOWER": "Power",
    "TATASTEEL": "Metals & Mining",
    "TECHM": "Information Technology",
    "TITAN": "Consumer Durables",
    "TORNTPHARM": "Healthcare",
    "TRENT": "Consumer Services",
    "TIINDIA": "Automobile and Auto Components",
    "UPL": "Chemicals",
    "ULTRACEMCO": "Construction Materials",
    "UNIONBANK": "Financial Services",
    "UNITDSPR": "Fast Moving Consumer Goods",
    "VBL": "Fast Moving Consumer Goods",
    "VEDL": "Metals & Mining",
    "VMM": "Consumer Services",
    "IDEA": "Telecommunication",
    "VOLTAS": "Consumer Durables",
    "WAAREEENER": "Capital Goods",
    "WIPRO": "Information Technology",
    "YESBANK": "Financial Services",
    "ZYDUSLIFE": "Healthcare",
    "MMM": "Industrials",
    "AOS": "Industrials",
    "ABT": "Health Care",
    "ABBV": "Health Care",
    "ACN": "Information Technology",
    "ADBE": "Information Technology",
    "AMD": "Information Technology",
    "AES": "Utilities",
    "AFL": "Financials",
    "A": "Health Care",
    "APD": "Materials",
    "ABNB": "Consumer Discretionary",
    "AKAM": "Information Technology",
    "ALB": "Materials",
    "ARE": "Real Estate",
    "ALGN": "Health Care",
    "ALLE": "Industrials",
    "LNT": "Utilities",
    "ALL": "Financials",
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "MO": "Consumer Staples",
    "AMZN": "Consumer Discretionary",
    "AMCR": "Materials",
    "AEE": "Utilities",
    "AEP": "Utilities",
    "AXP": "Financials",
    "AIG": "Financials",
    "AMT": "Real Estate",
    "AWK": "Utilities",
    "AMP": "Financials",
    "AME": "Industrials",
    "AMGN": "Health Care",
    "APH": "Information Technology",
    "ADI": "Information Technology",
    "AON": "Financials",
    "APA": "Energy",
    "APO": "Financials",
    "AAPL": "Information Technology",
    "AMAT": "Information Technology",
    "APP": "Information Technology",
    "APTV": "Consumer Discretionary",
    "ACGL": "Financials",
    "ADM": "Consumer Staples",
    "ARES": "Financials",
    "ANET": "Information Technology",
    "AJG": "Financials",
    "AIZ": "Financials",
    "T": "Communication Services",
    "ATO": "Utilities",
    "ADSK": "Information Technology",
    "ADP": "Industrials",
    "AZO": "Consumer Discretionary",
    "AVB": "Real Estate",
    "AVY": "Materials",
    "AXON": "Industrials",
    "BKR": "Energy",
    "BALL": "Materials",
    "BAC": "Financials",
    "BAX": "Health Care",
    "BDX": "Health Care",
    "BRK-B": "Financials",
    "BBY": "Consumer Discretionary",
    "TECH": "Health Care",
    "BIIB": "Health Care",
    "BLK": "Financials",
    "BX": "Financials",
    "XYZ": "Financials",
    "BNY": "Financials",
    "BA": "Industrials",
    "BKNG": "Consumer Discretionary",
    "BSX": "Health Care",
    "BMY": "Health Care",
    "AVGO": "Information Technology",
    "BR": "Industrials",
    "BRO": "Financials",
    "BF-B": "Consumer Staples",
    "BLDR": "Industrials",
    "BG": "Consumer Staples",
    "BXP": "Real Estate",
    "CHRW": "Industrials",
    "CDNS": "Information Technology",
    "CPT": "Real Estate",
    "CPB": "Consumer Staples",
    "COF": "Financials",
    "CAH": "Health Care",
    "CCL": "Consumer Discretionary",
    "CARR": "Industrials",
    "CVNA": "Consumer Discretionary",
    "CASY": "Consumer Staples",
    "CAT": "Industrials",
    "CBOE": "Financials",
    "CBRE": "Real Estate",
    "CDW": "Information Technology",
    "COR": "Health Care",
    "CNC": "Health Care",
    "CNP": "Utilities",
    "CF": "Materials",
    "CRL": "Health Care",
    "SCHW": "Financials",
    "CHTR": "Communication Services",
    "CVX": "Energy",
    "CMG": "Consumer Discretionary",
    "CB": "Financials",
    "CHD": "Consumer Staples",
    "CIEN": "Information Technology",
    "CI": "Health Care",
    "CINF": "Financials",
    "CTAS": "Industrials",
    "CSCO": "Information Technology",
    "C": "Financials",
    "CFG": "Financials",
    "CLX": "Consumer Staples",
    "CME": "Financials",
    "CMS": "Utilities",
    "KO": "Consumer Staples",
    "CTSH": "Information Technology",
    "COHR": "Information Technology",
    "COIN": "Financials",
    "CL": "Consumer Staples",
    "CMCSA": "Communication Services",
    "FIX": "Industrials",
    "CAG": "Consumer Staples",
    "COP": "Energy",
    "ED": "Utilities",
    "STZ": "Consumer Staples",
    "CEG": "Utilities",
    "COO": "Health Care",
    "CPRT": "Industrials",
    "GLW": "Information Technology",
    "CPAY": "Financials",
    "CTVA": "Materials",
    "CSGP": "Real Estate",
    "COST": "Consumer Staples",
    "CRH": "Materials",
    "CRWD": "Information Technology",
    "CCI": "Real Estate",
    "CSX": "Industrials",
    "CMI": "Industrials",
    "CVS": "Health Care",
    "DHR": "Health Care",
    "DRI": "Consumer Discretionary",
    "DDOG": "Information Technology",
    "DVA": "Health Care",
    "DECK": "Consumer Discretionary",
    "DE": "Industrials",
    "DELL": "Information Technology",
    "DAL": "Industrials",
    "DVN": "Energy",
    "DXCM": "Health Care",
    "FANG": "Energy",
    "DLR": "Real Estate",
    "DG": "Consumer Staples",
    "DLTR": "Consumer Staples",
    "D": "Utilities",
    "DPZ": "Consumer Discretionary",
    "DASH": "Consumer Discretionary",
    "DOV": "Industrials",
    "DOW": "Materials",
    "DHI": "Consumer Discretionary",
    "DTE": "Utilities",
    "DUK": "Utilities",
    "DD": "Materials",
    "ETN": "Industrials",
    "EBAY": "Consumer Discretionary",
    "SATS": "Communication Services",
    "ECL": "Materials",
    "EIX": "Utilities",
    "EW": "Health Care",
    "EA": "Communication Services",
    "ELV": "Health Care",
    "EME": "Industrials",
    "EMR": "Industrials",
    "ETR": "Utilities",
    "EOG": "Energy",
    "EPAM": "Information Technology",
    "EQT": "Energy",
    "EFX": "Industrials",
    "EQIX": "Real Estate",
    "EQR": "Real Estate",
    "ERIE": "Financials",
    "ESS": "Real Estate",
    "EL": "Consumer Staples",
    "EG": "Financials",
    "EVRG": "Utilities",
    "ES": "Utilities",
    "EXC": "Utilities",
    "EXE": "Energy",
    "EXPE": "Consumer Discretionary",
    "EXPD": "Industrials",
    "EXR": "Real Estate",
    "XOM": "Energy",
    "FFIV": "Information Technology",
    "FDS": "Financials",
    "FICO": "Information Technology",
    "FAST": "Industrials",
    "FRT": "Real Estate",
    "FDX": "Industrials",
    "FIS": "Financials",
    "FITB": "Financials",
    "FSLR": "Information Technology",
    "FE": "Utilities",
    "FISV": "Financials",
    "F": "Consumer Discretionary",
    "FTNT": "Information Technology",
    "FTV": "Industrials",
    "FOXA": "Communication Services",
    "FOX": "Communication Services",
    "BEN": "Financials",
    "FCX": "Materials",
    "GRMN": "Consumer Discretionary",
    "IT": "Information Technology",
    "GE": "Industrials",
    "GEHC": "Health Care",
    "GEV": "Industrials",
    "GEN": "Information Technology",
    "GNRC": "Industrials",
    "GD": "Industrials",
    "GIS": "Consumer Staples",
    "GM": "Consumer Discretionary",
    "GPC": "Consumer Discretionary",
    "GILD": "Health Care",
    "GPN": "Financials",
    "GL": "Financials",
    "GDDY": "Information Technology",
    "GS": "Financials",
    "HAL": "Energy",
    "HIG": "Financials",
    "HAS": "Consumer Discretionary",
    "HCA": "Health Care",
    "DOC": "Real Estate",
    "HSIC": "Health Care",
    "HSY": "Consumer Staples",
    "HPE": "Information Technology",
    "HLT": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "HON": "Industrials",
    "HRL": "Consumer Staples",
    "HST": "Real Estate",
    "HWM": "Industrials",
    "HPQ": "Information Technology",
    "HUBB": "Industrials",
    "HUM": "Health Care",
    "HBAN": "Financials",
    "HII": "Industrials",
    "IBM": "Information Technology",
    "IEX": "Industrials",
    "IDXX": "Health Care",
    "ITW": "Industrials",
    "INCY": "Health Care",
    "IR": "Industrials",
    "PODD": "Health Care",
    "INTC": "Information Technology",
    "IBKR": "Financials",
    "ICE": "Financials",
    "IFF": "Materials",
    "IP": "Materials",
    "INTU": "Information Technology",
    "ISRG": "Health Care",
    "IVZ": "Financials",
    "INVH": "Real Estate",
    "IQV": "Health Care",
    "IRM": "Real Estate",
    "JBHT": "Industrials",
    "JBL": "Information Technology",
    "JKHY": "Financials",
    "J": "Industrials",
    "JNJ": "Health Care",
    "JCI": "Industrials",
    "JPM": "Financials",
    "KVUE": "Consumer Staples",
    "KDP": "Consumer Staples",
    "KEY": "Financials",
    "KEYS": "Information Technology",
    "KMB": "Consumer Staples",
    "KIM": "Real Estate",
    "KMI": "Energy",
    "KKR": "Financials",
    "KLAC": "Information Technology",
    "KHC": "Consumer Staples",
    "KR": "Consumer Staples",
    "LHX": "Industrials",
    "LH": "Health Care",
    "LRCX": "Information Technology",
    "LVS": "Consumer Discretionary",
    "LDOS": "Industrials",
    "LEN": "Consumer Discretionary",
    "LII": "Industrials",
    "LLY": "Health Care",
    "LIN": "Materials",
    "LYV": "Communication Services",
    "LMT": "Industrials",
    "L": "Financials",
    "LOW": "Consumer Discretionary",
    "LULU": "Consumer Discretionary",
    "LITE": "Information Technology",
    "LYB": "Materials",
    "MTB": "Financials",
    "MPC": "Energy",
    "MAR": "Consumer Discretionary",
    "MRSH": "Financials",
    "MLM": "Materials",
    "MAS": "Industrials",
    "MA": "Financials",
    "MKC": "Consumer Staples",
    "MCD": "Consumer Discretionary",
    "MCK": "Health Care",
    "MDT": "Health Care",
    "MRK": "Health Care",
    "META": "Communication Services",
    "MET": "Financials",
    "MTD": "Health Care",
    "MGM": "Consumer Discretionary",
    "MCHP": "Information Technology",
    "MU": "Information Technology",
    "MSFT": "Information Technology",
    "MAA": "Real Estate",
    "MRNA": "Health Care",
    "TAP": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "MPWR": "Information Technology",
    "MNST": "Consumer Staples",
    "MCO": "Financials",
    "MS": "Financials",
    "MOS": "Materials",
    "MSI": "Information Technology",
    "MSCI": "Financials",
    "NDAQ": "Financials",
    "NTAP": "Information Technology",
    "NFLX": "Communication Services",
    "NEM": "Materials",
    "NWSA": "Communication Services",
    "NWS": "Communication Services",
    "NEE": "Utilities",
    "NKE": "Consumer Discretionary",
    "NI": "Utilities",
    "NDSN": "Industrials",
    "NSC": "Industrials",
    "NTRS": "Financials",
    "NOC": "Industrials",
    "NCLH": "Consumer Discretionary",
    "NRG": "Utilities",
    "NUE": "Materials",
    "NVDA": "Information Technology",
    "NVR": "Consumer Discretionary",
    "NXPI": "Information Technology",
    "ORLY": "Consumer Discretionary",
    "OXY": "Energy",
    "ODFL": "Industrials",
    "OMC": "Communication Services",
    "ON": "Information Technology",
    "OKE": "Energy",
    "ORCL": "Information Technology",
    "OTIS": "Industrials",
    "PCAR": "Industrials",
    "PKG": "Materials",
    "PLTR": "Information Technology",
    "PANW": "Information Technology",
    "PSKY": "Communication Services",
    "PH": "Industrials",
    "PAYX": "Industrials",
    "PYPL": "Financials",
    "PNR": "Industrials",
    "PEP": "Consumer Staples",
    "PFE": "Health Care",
    "PCG": "Utilities",
    "PM": "Consumer Staples",
    "PSX": "Energy",
    "PNW": "Utilities",
    "PNC": "Financials",
    "POOL": "Consumer Discretionary",
    "PPG": "Materials",
    "PPL": "Utilities",
    "PFG": "Financials",
    "PG": "Consumer Staples",
    "PGR": "Financials",
    "PLD": "Real Estate",
    "PRU": "Financials",
    "PEG": "Utilities",
    "PTC": "Information Technology",
    "PSA": "Real Estate",
    "PHM": "Consumer Discretionary",
    "PWR": "Industrials",
    "QCOM": "Information Technology",
    "DGX": "Health Care",
    "Q": "Information Technology",
    "RL": "Consumer Discretionary",
    "RJF": "Financials",
    "RTX": "Industrials",
    "O": "Real Estate",
    "REG": "Real Estate",
    "REGN": "Health Care",
    "RF": "Financials",
    "RSG": "Industrials",
    "RMD": "Health Care",
    "RVTY": "Health Care",
    "HOOD": "Financials",
    "ROK": "Industrials",
    "ROL": "Industrials",
    "ROP": "Information Technology",
    "ROST": "Consumer Discretionary",
    "RCL": "Consumer Discretionary",
    "SPGI": "Financials",
    "CRM": "Information Technology",
    "SNDK": "Information Technology",
    "SBAC": "Real Estate",
    "SLB": "Energy",
    "STX": "Information Technology",
    "SRE": "Utilities",
    "NOW": "Information Technology",
    "SHW": "Materials",
    "SPG": "Real Estate",
    "SWKS": "Information Technology",
    "SJM": "Consumer Staples",
    "SW": "Materials",
    "SNA": "Industrials",
    "SOLV": "Health Care",
    "SO": "Utilities",
    "LUV": "Industrials",
    "SWK": "Industrials",
    "SBUX": "Consumer Discretionary",
    "STT": "Financials",
    "STLD": "Materials",
    "STE": "Health Care",
    "SYK": "Health Care",
    "SMCI": "Information Technology",
    "SYF": "Financials",
    "SNPS": "Information Technology",
    "SYY": "Consumer Staples",
    "TMUS": "Communication Services",
    "TROW": "Financials",
    "TTWO": "Communication Services",
    "TPR": "Consumer Discretionary",
    "TRGP": "Energy",
    "TGT": "Consumer Staples",
    "TEL": "Information Technology",
    "TDY": "Information Technology",
    "TER": "Information Technology",
    "TSLA": "Consumer Discretionary",
    "TXN": "Information Technology",
    "TPL": "Energy",
    "TXT": "Industrials",
    "TMO": "Health Care",
    "TJX": "Consumer Discretionary",
    "TKO": "Communication Services",
    "TTD": "Communication Services",
    "TSCO": "Consumer Discretionary",
    "TT": "Industrials",
    "TDG": "Industrials",
    "TRV": "Financials",
    "TRMB": "Information Technology",
    "TFC": "Financials",
    "TYL": "Information Technology",
    "TSN": "Consumer Staples",
    "USB": "Financials",
    "UBER": "Industrials",
    "UDR": "Real Estate",
    "ULTA": "Consumer Discretionary",
    "UNP": "Industrials",
    "UAL": "Industrials",
    "UPS": "Industrials",
    "URI": "Industrials",
    "UNH": "Health Care",
    "UHS": "Health Care",
    "VLO": "Energy",
    "VEEV": "Health Care",
    "VTR": "Real Estate",
    "VLTO": "Industrials",
    "VRSN": "Information Technology",
    "VRSK": "Industrials",
    "VZ": "Communication Services",
    "VRTX": "Health Care",
    "VRT": "Industrials",
    "VTRS": "Health Care",
    "VICI": "Real Estate",
    "V": "Financials",
    "VST": "Utilities",
    "VMC": "Materials",
    "WRB": "Financials",
    "GWW": "Industrials",
    "WAB": "Industrials",
    "WMT": "Consumer Staples",
    "DIS": "Communication Services",
    "WBD": "Communication Services",
    "WM": "Industrials",
    "WAT": "Health Care",
    "WEC": "Utilities",
    "WFC": "Financials",
    "WELL": "Real Estate",
    "WST": "Health Care",
    "WDC": "Information Technology",
    "WY": "Real Estate",
    "WSM": "Consumer Discretionary",
    "WMB": "Energy",
    "WTW": "Financials",
    "WDAY": "Information Technology",
    "WYNN": "Consumer Discretionary",
    "XEL": "Utilities",
    "XYL": "Industrials",
    "YUM": "Consumer Discretionary",
    "ZBRA": "Information Technology",
    "ZBH": "Health Care",
    "ZTS": "Health Care",
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
    # Live NIFTY 200 fetch uses NSE's own "Industry" strings verbatim:
    "Financial Services": "🏦", "Information Technology": "💻",
    "Healthcare": "💊", "Oil Gas & Consumable Fuels": "🛢",
    "Metals & Mining": "🪙", "Fast Moving Consumer Goods": "🛒",
    "Automobile and Auto Components": "🚗", "Capital Goods": "⚙",
    "Construction": "🏗", "Construction Materials": "🧱",
    "Consumer Durables": "🛍", "Consumer Services": "🛍",
    "Services": "🛠", "Telecommunication": "📡", "Textiles": "🧵",
    # Live S&P 500 fetch uses Wikipedia's GICS Sector strings verbatim:
    "Health Care": "💊", "Communication Services": "📡",
    "Consumer Discretionary": "🛍", "Consumer Staples": "🛒",
    "Materials": "🪙", "Real Estate": "🏠", "Utilities": "💡",
}



# ---------------------------------------------------------------------------
# LIVE WATCHLIST — fetch the full NIFTY 200 / S&P 500 constituent lists at
# runtime instead of relying on the static lists above. Both fetchers return
# None on any failure (network error, unexpected/short file, source changed
# layout, etc.) so build_full_watchlist() can fall back to the static lists
# per-market — a network hiccup should never crash the whole run.
# ---------------------------------------------------------------------------

def fetch_nifty200_list():
    """Downloads the live NIFTY 200 constituent list from NSE's archives
    (columns: Company Name, Industry, Symbol, Series, ISIN Code). Returns a
    list of (ticker, sector_label) tuples, or None on failure."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
    req = urllib.request.Request(url, headers={
        # A plain urllib default User-Agent gets blocked by NSE's archives
        # host, so we ask for the CSV like a regular browser would.
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"),
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  [nifty200] live fetch failed ({e}) — using the static fallback list")
        return None

    try:
        reader = csv.DictReader(io.StringIO(raw))
        pairs = []
        for row in reader:
            symbol = (row.get("Symbol") or "").strip()
            industry = (row.get("Industry") or "").strip()
            if not symbol or symbol.upper().startswith("DUMMY"):
                # NSE lists a few "Dummy ..." placeholder rows around
                # demergers/corporate actions — these aren't real tickers.
                continue
            pairs.append((f"{symbol}.NS", industry or "Other"))
    except Exception as e:
        print(f"  [nifty200] couldn't parse live CSV ({e}) — using the static fallback list")
        return None

    if len(pairs) < 150:
        print(f"  [nifty200] only parsed {len(pairs)} rows (expected ~200) — using the static fallback list")
        return None

    print(f"  [nifty200] using {len(pairs)} live NIFTY 200 constituents")
    return pairs


def fetch_sp500_list():
    """Downloads the live S&P 500 constituent table from Wikipedia. Returns
    a list of (ticker, GICS_sector_label) tuples, or None on failure."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Wikipedia's first table on that page is the constituent list:
        # Symbol | Security | GICS Sector | GICS Sub-Industry | ...
        df = pd.read_html(url)[0]
        pairs = []
        for _, row in df.iterrows():
            # Yahoo Finance uses a hyphen where Wikipedia uses a dot for
            # dual-class tickers, e.g. BRK.B -> BRK-B, BF.B -> BF-B.
            symbol = str(row["Symbol"]).strip().replace(".", "-")
            gics = str(row["GICS Sector"]).strip()
            pairs.append((symbol, gics or "Other"))
    except Exception as e:
        print(f"  [sp500] live fetch failed ({e}) — using the static fallback list")
        return None

    if len(pairs) < 400:
        print(f"  [sp500] only parsed {len(pairs)} rows (expected ~500) — using the static fallback list")
        return None

    # Belt-and-braces: make sure PLTR is present even if Wikipedia's table
    # layout hiccups on a given run.
    if not any(t == "PLTR" for t, _ in pairs):
        pairs.append(("PLTR", "Information Technology"))

    print(f"  [sp500] using {len(pairs)} live S&P 500 constituents")
    return pairs


def build_full_watchlist():
    """Builds the final ticker universe to scan, plus a sector-label dict to
    merge into SECTOR_MAP, using live NIFTY 200 / S&P 500 data where
    available (and enabled) and falling back to the static lists per-market
    otherwise. The 7 sector ETFs (NIFTYBEES etc.) are always appended since
    they're a deliberate addition, not official index constituents."""
    watchlist = []
    sector_updates = {}

    live = fetch_nifty200_list() if USE_FULL_NIFTY200 else None
    if live:
        for ticker, sector in live:
            watchlist.append(ticker)
            sector_updates[ticker.replace(".NS", "")] = sector
    else:
        watchlist.extend(INDIA_WATCHLIST_STATIC)

    # Always include the sector ETFs, even on a live-fetched run — they
    # aren't official NIFTY 200 constituents so wouldn't otherwise appear.
    etf_tickers = [t for t in INDIA_WATCHLIST_STATIC
                   if SECTOR_MAP.get(t.replace(".NS", "")) == "ETF"]
    for t in etf_tickers:
        if t not in watchlist:
            watchlist.append(t)

    live_us = fetch_sp500_list() if USE_FULL_SP500 else None
    if live_us:
        for ticker, sector in live_us:
            watchlist.append(ticker)
            sector_updates[ticker] = sector
    else:
        watchlist.extend(USA_WATCHLIST_STATIC)

    # De-duplicate while preserving order (a ticker could in principle
    # appear in both a live list and a leftover static entry).
    seen = set()
    deduped = []
    for t in watchlist:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped, sector_updates


# Module-level default so anything importing this file (or --debug) still
# has a usable watchlist without needing to call main() first. main()
# overwrites this with the full live universe before scanning.
WATCHLIST = list(INDIA_WATCHLIST_STATIC)


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


def _fib_zigzag_pivots(df: pd.DataFrame, pct: float):
    """
    Wick-based zigzag (ported from fib_swing_trade_plan.py): walks the
    High/Low range and only confirms a new pivot once price has reversed
    by at least `pct`% from the running extreme since the last confirmed
    pivot. Peaks are tracked off High, troughs off Low. Returns a list of
    (position, price, kind) tuples in chronological order, kind = 'H'/'L'.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(highs)
    if n < 2:
        return []

    pivots = []
    trend = None
    ext_pos, ext_val = 0, highs[0]
    run_max_pos, run_max_val = 0, highs[0]
    run_min_pos, run_min_val = 0, lows[0]

    for i in range(1, n):
        hi, lo = highs[i], lows[i]
        if trend is None:
            if hi > run_max_val:
                run_max_pos, run_max_val = i, hi
            if lo < run_min_val:
                run_min_pos, run_min_val = i, lo

            if lo <= run_max_val * (1 - pct / 100):
                trend = "down"
                pivots.append((run_max_pos, run_max_val, "H"))
                ext_pos, ext_val = i, lo
            elif hi >= run_min_val * (1 + pct / 100):
                trend = "up"
                pivots.append((run_min_pos, run_min_val, "L"))
                ext_pos, ext_val = i, hi
        elif trend == "up":
            if hi > ext_val:
                ext_pos, ext_val = i, hi
            elif lo <= ext_val * (1 - pct / 100):
                pivots.append((ext_pos, ext_val, "H"))
                trend = "down"
                ext_pos, ext_val = i, lo
        else:  # trend == "down"
            if lo < ext_val:
                ext_pos, ext_val = i, lo
            elif hi >= ext_val * (1 + pct / 100):
                pivots.append((ext_pos, ext_val, "L"))
                trend = "up"
                ext_pos, ext_val = i, hi

    return pivots


def _fib_detect_last_swing(df: pd.DataFrame):
    """
    Ported from fib_swing_trade_plan.py's detect_last_swing() (SWING_MODE
    fixed to "recent" here — the freshest completed leg, which is what
    matters for "where's the Fib range right now" on a stock that just
    fired an RSI recovery signal). Returns (swing_high, swing_high_date,
    swing_low, swing_low_date, trend, is_extended).
    """
    pivots = _fib_zigzag_pivots(df, FIB_ZIGZAG_PCT)
    highs_all = df["High"].values
    lows_all = df["Low"].values
    is_extended = False

    if len(pivots) >= 2:
        pos_a, val_a, kind_a = pivots[-2]
        pos_b, val_b, kind_b = pivots[-1]

        is_latest_leg = (pos_b == pivots[-1][0])
        if is_latest_leg:
            if kind_b == "L":
                tail = highs_all[pos_b:]
                run_rel = int(np.argmax(tail))
            else:  # kind_b == "H"
                tail = lows_all[pos_b:]
                run_rel = int(np.argmin(tail))
            run_pos = pos_b + run_rel
            run_val = float(tail[run_rel])

            breaks_out = (run_val > val_a) if kind_b == "L" else (run_val < val_a)
            if run_pos > pos_b and breaks_out:
                pos_a, val_a, kind_a = pos_b, val_b, kind_b
                pos_b, val_b, kind_b = run_pos, run_val, ("H" if kind_b == "L" else "L")
                is_extended = True
    else:
        n = len(highs_all)
        high_idx, low_idx = [], []
        for i in range(1, n - 1):
            radius = min(FIB_SWING_ORDER, i, n - 1 - i)
            hi_window = highs_all[i - radius:i + radius + 1]
            lo_window = lows_all[i - radius:i + radius + 1]
            if highs_all[i] == hi_window.max():
                high_idx.append(i)
            if lows_all[i] == lo_window.min():
                low_idx.append(i)
        if high_idx and low_idx:
            pos_a, val_a, kind_a = high_idx[-1], float(highs_all[high_idx[-1]]), "H"
            pos_b, val_b, kind_b = low_idx[-1], float(lows_all[low_idx[-1]]), "L"
        else:
            hi_pos, lo_pos = int(highs_all.argmax()), int(lows_all.argmin())
            pos_a, val_a, kind_a = hi_pos, float(highs_all[hi_pos]), "H"
            pos_b, val_b, kind_b = lo_pos, float(lows_all[lo_pos]), "L"

    if kind_a == "H":
        hi_pos, swing_high = pos_a, val_a
        lo_pos, swing_low = pos_b, val_b
    else:
        hi_pos, swing_high = pos_b, val_b
        lo_pos, swing_low = pos_a, val_a

    swing_high_date = str(df.index[hi_pos].date())
    swing_low_date = str(df.index[lo_pos].date())
    trend = "Uptrend" if lo_pos < hi_pos else "Downtrend"

    return swing_high, swing_high_date, swing_low, swing_low_date, trend, is_extended


def _fib_zone(current_price: float, levels: dict, trend: str) -> str:
    """Which Fib retracement band `current_price` currently sits in (ported
    from fib_swing_trade_plan.py's compute_fib_zone)."""
    if not levels:
        return "—"
    ratios = sorted(levels.keys())
    prices = [levels[r] for r in ratios]

    if trend == "Uptrend":
        if current_price >= prices[0]:
            return "Above 0% (new high)"
        if current_price <= prices[-1]:
            return "Beyond 100% (retraced)"
        for i in range(len(ratios) - 1):
            hi_p, lo_p = prices[i], prices[i + 1]
            if lo_p <= current_price <= hi_p:
                return f"{ratios[i] * 100:.1f}-{ratios[i + 1] * 100:.1f}%"
    else:
        if current_price <= prices[0]:
            return "Below 0% (new low)"
        if current_price >= prices[-1]:
            return "Beyond 100% (retraced)"
        for i in range(len(ratios) - 1):
            lo_p, hi_p = prices[i], prices[i + 1]
            if lo_p <= current_price <= hi_p:
                return f"{ratios[i] * 100:.1f}-{ratios[i + 1] * 100:.1f}%"
    return "—"


def compute_fib_plan(df: pd.DataFrame, current_price: float) -> dict | None:
    """
    Fibonacci context for a stock that already passed the RSI recovery
    signal — reuses the same OHLC `df` the RSI scan fetched (no extra
    network call). Returns None if there isn't enough history or no valid
    swing range can be established; otherwise a dict with the swing,
    retracement levels, current Fib zone, and an entry/stop/target plan
    (same formulas as fib_swing_trade_plan.py).
    """
    if len(df) < FIB_MIN_BARS or "High" not in df.columns or "Low" not in df.columns:
        return None

    try:
        swing_high, sh_date, swing_low, sl_date, trend, is_extended = _fib_detect_last_swing(df)
    except Exception:
        return None

    if swing_high <= swing_low:
        return None

    diff = swing_high - swing_low
    levels = {r: 0.0 for r in FIB_RATIOS}

    if trend == "Uptrend":
        for r in FIB_RATIOS:
            levels[r] = swing_high - diff * r
        extension_target = swing_low + diff * FIB_EXTENSION
        entry_high = levels[0.5]
        entry_low = levels[0.618]
        stop_loss = levels[0.786]
        tp1 = swing_high
        tp2 = extension_target
    else:  # Downtrend
        for r in FIB_RATIOS:
            levels[r] = swing_low + diff * r
        extension_target = swing_high - diff * FIB_EXTENSION
        entry_high = levels[0.618]
        entry_low = levels[0.5]
        stop_loss = levels[0.786]
        tp1 = swing_low
        tp2 = extension_target

    return {
        "fib_trend": trend,
        "fib_extended": is_extended,
        "swing_high": round(float(swing_high), 2),
        "swing_low": round(float(swing_low), 2),
        "swing_high_date": sh_date,
        "swing_low_date": sl_date,
        "fib_zone": _fib_zone(current_price, levels, trend),
        "entry_low": round(float(entry_low), 2),
        "entry_high": round(float(entry_high), 2),
        "stop_loss": round(float(stop_loss), 2),
        "tp1": round(float(tp1), 2),
        "tp2": round(float(tp2), 2),
    }


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


def fetch_multi_timeframe_parallel(tickers: list, specs: dict,
                                    max_workers: int = 10) -> dict:
    """Fetch several timeframes for all tickers through ONE shared thread
    pool, instead of one all-tickers-Daily pass followed by a separate
    all-tickers-15min pass. Every (ticker, timeframe) request is queued as
    its own job up front, so a single stock's Daily and 15-Min requests
    can genuinely be in flight to Yahoo at the same moment -- not just
    "parallel across stocks", but parallel across stocks AND timeframes.

    specs: {tf_key: (period, interval)}, e.g. {"1d": ("6mo", "1d"),
    "15m": ("60d", "15m")}.
    Returns {tf_key: {ticker: DataFrame}}.

    This doesn't change how much total data is pulled -- same number of
    requests as before -- it just lets them overlap in time instead of
    running in two back-to-back batches, which cuts wall-clock time
    further. Nothing about this depends on where the script is hosted or
    run from (local machine, a GitHub Actions runner, etc.) -- the
    concurrency is purely a property of this process talking to Yahoo, so
    it behaves the same after you push this to GitHub.
    """
    out = {tf_key: {} for tf_key in specs}
    jobs = [(ticker, tf_key) for ticker in tickers for tf_key in specs]
    total = len(jobs)
    done = 0

    def _fetch_one(ticker, tf_key):
        period, interval = specs[tf_key]
        try:
            return ticker, tf_key, yf.Ticker(ticker).history(period=period, interval=interval)
        except Exception:
            return ticker, tf_key, pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_one, ticker, tf_key) for ticker, tf_key in jobs]
        for fut in concurrent.futures.as_completed(futures):
            ticker, tf_key, df = fut.result()
            out[tf_key][ticker] = df
            done += 1
            print(f"\rFetching Daily + 15-Min data... [{done}/{total}]", end="", flush=True)
    print("\r" + " " * 60 + "\r", end="", flush=True)
    return out


def resample_15m_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Derive approximate 1-hour bars from already-fetched 15-minute bars
    instead of making a second network fetch. 15m and 1h use the exact
    same period="60d" lookback window, so this eliminates 1/3 of all
    requests for free. Caveat: Yahoo's native 1h bars are aligned to each
    exchange's session open (e.g. 9:15 for NSE), while a plain resample
    aligns to clock hours -- so RSI values here will be close to, but not
    bit-for-bit identical to, a native 1h fetch. Fine for a screener;
    fetch 1h natively instead if you need exact parity."""
    if df_15m is None or df_15m.empty:
        return pd.DataFrame()
    agg = {c: a for c, a in
           {"Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"}.items()
           if c in df_15m.columns}
    out = df_15m.resample("1h", label="right", closed="right").agg(agg)
    return out.dropna(subset=["Close"])


def scan(verbose: bool = False, timeframe: str = DEFAULT_TIMEFRAME,
          prefetched: dict = None) -> tuple[list[dict], dict]:
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

    If `prefetched` is given ({ticker: DataFrame}), no network calls are
    made here at all -- the data was already fetched (in parallel, or via
    the 15m->1h resample) before scan() was called. Falls back to the old
    one-ticker-at-a-time fetch if prefetched is None.
    """
    tf = TIMEFRAMES[timeframe]
    results = []
    all_rsi = {}
    total = len(WATCHLIST)

    for idx, ticker in enumerate(WATCHLIST, 1):
        if verbose:
            print(f"[{idx}/{total}] {ticker} ...", end=" ", flush=True)
        else:
            print(f"\rScanning ({tf['label']})... [{idx}/{total}] {ticker:<16}", end="", flush=True)
        try:
            if prefetched is not None:
                df = prefetched.get(ticker, pd.DataFrame())
            else:
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
                fib = compute_fib_plan(df, last_close)  # informational only — never affects the signal itself
                results.append({
                    "symbol": ticker.replace(".NS", ""),
                    "market": "India" if ticker.endswith(".NS") else "USA",
                    "ltp": last_close,
                    "pct_chg": pct_chg,
                    **signal,
                    "fib": fib,
                })
                if verbose:
                    print(f"SIGNAL (RSI {signal['current_rsi']}, trough {signal['trough_rsi']}, sma9 {signal['rsi_sma']}, sma21 {signal['rsi_long_sma']})")
            elif verbose:
                print("no signal")
        except Exception as e:
            if verbose:
                print(f"error: {e}")
            continue
        finally:
            # Only needed for the old serial fetch path -- when data was
            # prefetched in parallel up front, there's no per-ticker network
            # call happening in this loop anymore, so no need to throttle it.
            if prefetched is None and idx < total:
                time.sleep(0.2)

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
    market_counts = {"India": 0, "USA": 0}

    if results:
        for r in results:
            sector = SECTOR_MAP.get(r["symbol"], "Other")
            chg_class = "pos" if r["pct_chg"] >= 0 else "neg"
            rsi_color = _rsi_color(r["current_rsi"])
            trend_label, trend_class = _rsi_trend_chip(r["current_rsi"], r["rsi_sma"], r["rsi_long_sma"])
            trend_counts[trend_label] = trend_counts.get(trend_label, 0) + 1
            market = r.get("market", "India")
            market_counts[market] = market_counts.get(market, 0) + 1
            spark = _sparkline_svg([r["trough_rsi"], r["rsi_long_sma"], r["rsi_sma"], r["current_rsi"]])

            fib = r.get("fib")
            if fib:
                fib_trend_class = "chip-up" if fib["fib_trend"] == "Uptrend" else "chip-down"
                ext_note = " *" if fib["fib_extended"] else ""
                fib_trend_cell = f'<span class="chip {fib_trend_class}">{fib["fib_trend"]}{ext_note}</span>'
                fib_zone_cell = fib["fib_zone"]
                fib_swing_cell = f'H {fib["swing_high"]} / L {fib["swing_low"]}'
                fib_entry_cell = f'{fib["entry_low"]} - {fib["entry_high"]}'
                fib_stop_cell = f'{fib["stop_loss"]}'
                fib_target_cell = f'{fib["tp1"]} / {fib["tp2"]}'
                fib_zone_attr = fib["fib_zone"]
                fib_trend_attr = fib["fib_trend"]
                fib_swing_high_attr = fib["swing_high"]
                fib_entry_low_attr = fib["entry_low"]
                fib_stop_attr = fib["stop_loss"]
                fib_tp1_attr = fib["tp1"]
            else:
                fib_trend_cell = fib_zone_cell = fib_swing_cell = "—"
                fib_entry_cell = fib_stop_cell = fib_target_cell = "—"
                fib_zone_attr = "—"
                fib_trend_attr = ""
                fib_swing_high_attr = fib_entry_low_attr = fib_stop_attr = fib_tp1_attr = ""

            rows += f"""
            <tr data-symbol="{r['symbol']}" data-sector="{sector}" data-trend="{trend_label}"
                data-market="{market}"
                data-rsi="{r['current_rsi']}" data-recovery="{r['recovery_points']}"
                data-trough-age="{r['bars_since_trough']}" data-chg="{r['pct_chg']}"
                data-fibzone="{fib_zone_attr}" data-fib-trend="{fib_trend_attr}"
                data-swing-high="{fib_swing_high_attr}" data-entry-low="{fib_entry_low_attr}"
                data-stop="{fib_stop_attr}" data-tp1="{fib_tp1_attr}">
                <td class="sym">{r['symbol']}<span class="sector-tag">{sector}</span><span class="market-tag market-{market.lower()}">{market}</span></td>
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
                <td>{fib_trend_cell}</td>
                <td>{fib_swing_cell}</td>
                <td>{fib_zone_cell}</td>
                <td>{fib_entry_cell}</td>
                <td>{fib_stop_cell}</td>
                <td>{fib_target_cell}</td>
            </tr>"""
    else:
        rows = '<tr><td colspan="17" class="empty">No stocks currently match the RSI band-recovery criteria.</td></tr>'

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
    market_tabs_html = f"""
        <div class="market-tabs" role="tablist" aria-label="Market filter">
            <button class="market-tab active" data-market="all" role="tab" aria-selected="true">All Markets <span class="market-tab-count">{total_signals}</span></button>
            <button class="market-tab" data-market="India" role="tab" aria-selected="false">🇮🇳 India <span class="market-tab-count">{market_counts['India']}</span></button>
            <button class="market-tab" data-market="USA" role="tab" aria-selected="false">🇺🇸 USA <span class="market-tab-count">{market_counts['USA']}</span></button>
        </div>"""
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
        {market_tabs_html}
        {trend_tabs_html}
        <div class="table-scroll">
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
                    <th data-key="trend-label" data-type="text">Trend</th>
                    <th data-key="recovery" data-type="num">Recovery</th>
                    <th data-key="fib-trend" data-type="text">Fib Trend</th>
                    <th data-key="swing-high" data-type="num">Swing (H/L)</th>
                    <th data-key="fibzone" data-type="text">Fib Zone</th>
                    <th data-key="entry-low" data-type="num">Entry Zone</th>
                    <th data-key="stop" data-type="num">Stop</th>
                    <th data-key="tp1" data-type="num">Target (TP1/TP2)</th>
                </tr>
            </thead>
            <tbody>{rows}
                <tr class="no-sector-match" style="display:none">
                    <td colspan="17" class="empty">No signals match the current filter(s). Adjust or clear the sector/trend filter above.</td>
                </tr>
            </tbody>
        </table>
        </div>
    </div>"""

    return panel_html, len(results)


def build_html(results_by_tf: dict[str, list], all_rsi_by_tf: dict[str, dict]) -> str:
    # Report timestamp is always shown in US Central time (CST/CDT, whichever
    # is currently in effect), regardless of what timezone the machine
    # running this script is actually in -- so the report is consistent
    # whether it's generated locally or from a CI runner in another region.
    now_central = datetime.datetime.now(datetime.timezone.utc).astimezone(ZoneInfo("America/Chicago"))
    ts = now_central.strftime("%Y-%m-%d %I:%M:%S %p %Z")
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
    html {{ width: 100%; }}
    body {{
        background: var(--bg); color: var(--text);
        font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
        margin: 0; padding: 32px clamp(16px, 3vw, 40px);
        width: 100%; min-height: 100vh;
    }}
    .wrap {{ max-width: 1900px; width: 100%; margin: 0 auto; }}
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

    .table-scroll {{
        width: 100%; overflow-x: auto; border-radius: 10px;
        -webkit-overflow-scrolling: touch; scrollbar-width: thin;
    }}
    .table-scroll::-webkit-scrollbar {{ height: 8px; }}
    .table-scroll::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
    .table-scroll::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

    table {{ width: 100%; min-width: 1500px; border-collapse: collapse; background: var(--panel);
        border: 1px solid var(--border); border-radius: 10px; }}
    th {{ text-align: left; font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--muted); padding: 10px 12px;
        border-bottom: 1px solid var(--border); cursor: pointer; user-select: none;
        white-space: nowrap; position: sticky; top: 0; background: var(--panel); z-index: 1; }}
    th:hover {{ color: var(--text); }}
    th.sorted::after {{ content: " \\2195"; color: var(--accent); }}
    td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 13px; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    .sym {{ font-weight: 600; }}
    .sector-tag {{ display: inline-block; margin-left: 8px; font-size: 10px; font-weight: 500;
        color: var(--muted); background: var(--panel2); border: 1px solid var(--border);
        border-radius: 5px; padding: 1px 6px; text-transform: uppercase; letter-spacing: 0.03em; }}
    .market-tag {{ display: inline-block; margin-left: 6px; font-size: 10px; font-weight: 600;
        border-radius: 5px; padding: 1px 6px; }}
    .market-tag.market-india {{ color: #ff9933; background: rgba(255,153,51,0.14); border: 1px solid rgba(255,153,51,0.35); }}
    .market-tag.market-usa {{ color: #5b9bff; background: rgba(91,155,255,0.14); border: 1px solid rgba(91,155,255,0.35); }}
    .pos {{ color: var(--pos); }}
    .neg {{ color: var(--neg); }}
    .rsi-badge {{ font-weight: 700; padding: 2px 9px; border-radius: 999px; font-size: 12.5px; }}
    .chip {{ padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.03em; }}
    .chip-up {{ background: rgba(38,208,124,0.14); color: var(--pos); }}
    .chip-down {{ background: rgba(255,84,112,0.14); color: var(--neg); }}
    .chip-neutral {{ background: rgba(139,147,167,0.14); color: var(--muted); }}

    /* -- Market filter tabs (All Markets / India / USA) -- */
    .market-tabs {{
        display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 4px;
    }}
    .market-tab {{
        display: inline-flex; align-items: center; gap: 6px;
        background: var(--panel2); border: 1px solid var(--border); color: var(--muted);
        border-radius: 999px; padding: 6px 14px; font-size: 12px; font-weight: 600;
        cursor: pointer; transition: border-color 0.15s, color 0.15s, background 0.15s;
    }}
    .market-tab:hover {{ border-color: var(--accent); color: var(--text); }}
    .market-tab-count {{
        font-size: 10px; font-weight: 700; color: var(--muted);
        background: rgba(139,147,167,0.14); border-radius: 999px; padding: 1px 7px;
    }}
    .market-tab.active {{ color: var(--text); border-color: var(--accent); background: rgba(0,229,199,0.08); }}
    .market-tab.active .market-tab-count {{ color: var(--accent); background: rgba(0,229,199,0.14); }}
    .market-tab[data-market="India"].active {{ border-color: #ff9933; background: rgba(255,153,51,0.10); }}
    .market-tab[data-market="India"].active .market-tab-count {{ color: #ff9933; background: rgba(255,153,51,0.18); }}
    .market-tab[data-market="USA"].active {{ border-color: #5b9bff; background: rgba(91,155,255,0.10); }}
    .market-tab[data-market="USA"].active .market-tab-count {{ color: #5b9bff; background: rgba(91,155,255,0.18); }}

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

    /* -- Responsive breakpoints -- */
    @media (max-width: 900px) {{
        body {{ padding: 16px; }}
        .hdr-bottom {{ padding: 10px 12px; }}
        .ticker-grid {{ grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); }}
        table {{ min-width: 1200px; }}
    }}
    @media (max-width: 520px) {{
        h1 {{ font-size: 17px; }}
        .sub {{ font-size: 11px; }}
        table {{ min-width: 1000px; }}
        th, td {{ padding: 8px 9px; font-size: 12px; }}
    }}
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
    const panelActiveMarket = {{}};

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
            'trend-label': () => row.dataset.trend,
            recovery: () => parseFloat(row.dataset.recovery),
            'fib-trend': () => row.dataset.fibTrend,
            'swing-high': () => parseFloat(row.dataset.swingHigh),
            fibzone: () => row.dataset.fibzone,
            'entry-low': () => parseFloat(row.dataset.entryLow),
            stop: () => parseFloat(row.dataset.stop),
            tp1: () => parseFloat(row.dataset.tp1),
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

    // ---- Combined sector + trend + market filtering ----
    function applyFilters(tf) {{
        const {{ noSectorMatchRow }} = panelEls(tf);
        const sector = panelActiveSector[tf];
        const trend = panelActiveTrend[tf];
        const market = panelActiveMarket[tf];
        const rows = dataRows(tf);
        let visible = 0;
        rows.forEach(r => {{
            const sectorOk = !sector || r.dataset.sector === sector;
            const trendOk = !trend || trend === 'all' || r.dataset.trend === trend;
            const marketOk = !market || market === 'all' || r.dataset.market === market;
            const match = sectorOk && trendOk && marketOk;
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

    function setActiveMarket(tf, market, btnEl) {{
        panelActiveMarket[tf] = market === 'all' ? null : market;
        const panel = document.querySelector('.tf-panel[data-tf="' + tf + '"]');
        const tabs = panel.querySelectorAll('.market-tab');
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
        panelActiveMarket[tf] = null;
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
        const marketTabs = panel.querySelectorAll('.market-tab');
        marketTabs.forEach(btn => {{
            btn.addEventListener('click', () => setActiveMarket(tf, btn.dataset.market, btn));
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
    """Print a full pass/fail breakdown of every check for a single ticker.
    Tries the ticker as given first (covers bare USA tickers like AAPL or
    PLTR), then falls back to appending .NS for bare NSE symbols (e.g.
    "INFY") if the as-given form returns no data."""
    ticker = ticker.upper()
    candidates = [ticker] if ticker.endswith(".NS") else [ticker, ticker + ".NS"]
    tf = TIMEFRAMES[timeframe]

    df = pd.DataFrame()
    for cand in candidates:
        print(f"Fetching {cand} ({tf['label']}) ...")
        d = yf.Ticker(cand).history(period=tf["period"], interval=tf["interval"])
        if not d.empty:
            df = d
            ticker = cand
            break
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

    global WATCHLIST
    print("Building company universe (live NIFTY 200 / S&P 500 lookups)...")
    WATCHLIST, sector_updates = build_full_watchlist()
    SECTOR_MAP.update(sector_updates)

    # Fetch Daily and 15-Min data for the whole watchlist through ONE
    # shared thread pool, so a given stock's Daily and 15-Min requests can
    # be in flight to Yahoo at the same time, not just parallel-across-
    # stocks in two back-to-back batches. 1-Hour bars are then DERIVED
    # from the 15-Min data (same 60-day window) instead of fetched
    # separately -- so total network traffic is ~2x710 requests instead
    # of ~3x710, and those requests overlap as much as MAX_FETCH_WORKERS
    # allows. Tune MAX_FETCH_WORKERS down (e.g. 6) if you start seeing
    # rate-limit errors, or up (e.g. 16-20) if it's stable and you want
    # it faster still -- this applies the same whether you run it locally
    # or from a GitHub Actions workflow.
    MAX_FETCH_WORKERS = 10
    print(f"Fetching Daily + 15-Min data for {len(WATCHLIST)} stocks (parallel, combined)...")
    fetched = fetch_multi_timeframe_parallel(
        WATCHLIST,
        specs={
            "1d": (TIMEFRAMES["1d"]["period"], TIMEFRAMES["1d"]["interval"]),
            "15m": (TIMEFRAMES["15m"]["period"], TIMEFRAMES["15m"]["interval"]),
        },
        max_workers=MAX_FETCH_WORKERS,
    )
    daily_data = fetched["1d"]
    m15_data = fetched["15m"]

    print("Deriving 1-Hour bars from 15-Min data (no extra requests)...")
    h1_data = {t: resample_15m_to_1h(df) for t, df in m15_data.items()}

    prefetched_by_tf = {"1d": daily_data, "1h": h1_data, "15m": m15_data}

    results_by_tf = {}
    all_rsi_by_tf = {}
    for tf_key, tf in TIMEFRAMES.items():
        print(f"Scanning {len(WATCHLIST)} stocks for RSI band recovery ({tf['label']}, down->up, 30-55)...")
        results, all_rsi = scan(verbose=verbose, timeframe=tf_key,
                                 prefetched=prefetched_by_tf[tf_key])
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
