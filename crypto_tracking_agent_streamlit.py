import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ------------------------------
# Simple Crypto Tracking Agent
# ------------------------------
# MVP purpose:
# - Track selected wallets on EVM chains
# - Pull recent token transfers from Moralis
# - Infer simple BUY/SELL signals from inbound/outbound transfers
# - Enrich tokens with DEX Screener market data
# - Surface repeated buys across tracked wallets
#
# Notes:
# - This is a research dashboard, not auto-trading software.
# - Requires a Moralis API key for wallet transfer data.
# - DEX Screener enrichment uses public endpoints.

APP_TITLE = "Crypto Tracking Agent"
APP_SUBTITLE = "Track wallets, spot repeated buys, and review token momentum"
MORALIS_BASE = "https://deep-index.moralis.io/api/v2.2"
REQUEST_TIMEOUT = 20
DEFAULT_LOOKBACK_HOURS = 48
DEFAULT_LIMIT = 30
SUPPORTED_CHAINS = {
    "eth": "Ethereum",
    "bsc": "BNB Chain",
    "polygon": "Polygon",
    "base": "Base",
    "arbitrum": "Arbitrum",
    "optimism": "Optimism",
    "avalanche": "Avalanche",
}
DEXSCREENER_CHAIN_MAP = {
    "eth": "ethereum",
    "bsc": "bsc",
    "polygon": "polygon",
    "base": "base",
    "arbitrum": "arbitrum",
    "optimism": "optimism",
    "avalanche": "avalanche",
}
STABLE_SYMBOLS = {"USDT", "USDC", "DAI", "BUSD", "FDUSD", "TUSD", "USDE", "USDC.E", "USDT.E"}
IGNORE_SYMBOLS = STABLE_SYMBOLS.union({"WETH", "WBTC", "ETH", "BNB", "MATIC", "AVAX", "OP", "ARB"})


# ------------------------------
# Helpers
# ------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_num(value: Any) -> str:
    try:
        num = float(value)
    except Exception:
        return "-"
    if abs(num) >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"{num/1_000:.2f}K"
    if abs(num) >= 1:
        return f"{num:.2f}"
    return f"{num:.6f}"


def parse_wallets(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
        else:
            parts = [p.strip() for p in line.split()]
        if len(parts) == 2:
            label, address = parts
            chain = "eth"
        elif len(parts) >= 3:
            label, address, chain = parts[0], parts[1], parts[2].lower()
        else:
            continue
        if chain not in SUPPORTED_CHAINS:
            chain = "eth"
        rows.append({"label": label, "address": address.lower(), "chain": chain})
    return rows


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


# ------------------------------
# API Calls
# ------------------------------
@st.cache_data(ttl=60)
def get_wallet_token_transfers(
    address: str,
    chain: str,
    api_key: str,
    hours_back: int,
    limit: int,
) -> List[Dict[str, Any]]:
    to_date = now_utc()
    from_date = to_date - timedelta(hours=hours_back)
    url = f"{MORALIS_BASE}/{address}/erc20/transfers"
    headers = {"X-API-Key": api_key}
    params = {
        "chain": chain,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "limit": max(1, min(limit, 100)),
        "order": "DESC",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Moralis error {resp.status_code}: {resp.text[:250]}")
    data = resp.json()
    return data.get("result", []) or []


@st.cache_data(ttl=120)
def get_token_market_snapshot(chain: str, token_address: str) -> Dict[str, Any]:
    ds_chain = DEXSCREENER_CHAIN_MAP.get(chain, chain)
    url = f"https://api.dexscreener.com/token-pairs/v1/{ds_chain}/{token_address}"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        return {}
    pairs = resp.json() or []
    if not isinstance(pairs, list) or not pairs:
        return {}

    def rank_pair(pair: Dict[str, Any]) -> Tuple[float, float]:
        liquidity = float(safe_get(pair, "liquidity", "usd", default=0) or 0)
        volume = float(safe_get(pair, "volume", "h24", default=0) or 0)
        return liquidity, volume

    best = sorted(pairs, key=rank_pair, reverse=True)[0]
    return {
        "priceUsd": safe_get(best, "priceUsd"),
        "liquidityUsd": safe_get(best, "liquidity", "usd"),
        "fdv": best.get("fdv"),
        "marketCap": best.get("marketCap"),
        "volume24h": safe_get(best, "volume", "h24"),
        "priceChangeM5": safe_get(best, "priceChange", "m5"),
        "priceChangeH1": safe_get(best, "priceChange", "h1"),
        "priceChangeH6": safe_get(best, "priceChange", "h6"),
        "priceChangeH24": safe_get(best, "priceChange", "h24"),
        "pairUrl": best.get("url"),
        "dexId": best.get("dexId"),
        "pairAddress": best.get("pairAddress"),
    }


# ------------------------------
# Signal Logic
# ------------------------------
def infer_action(transfer: Dict[str, Any], wallet: str) -> str:
    from_addr = str(transfer.get("from_address") or "").lower()
    to_addr = str(transfer.get("to_address") or "").lower()
    wallet = wallet.lower()
    if to_addr == wallet and from_addr != wallet:
        return "BUY"
    if from_addr == wallet and to_addr != wallet:
        return "SELL"
    return "MOVE"


def normalize_transfer(transfer: Dict[str, Any], wallet_meta: Dict[str, str]) -> Dict[str, Any]:
    decimals = int(transfer.get("token_decimals") or 0)
    raw_value = transfer.get("value") or "0"
    try:
        amount = int(raw_value) / (10 ** decimals) if decimals >= 0 else 0
    except Exception:
        amount = 0

    symbol = (transfer.get("token_symbol") or "?").upper()
    token_address = (transfer.get("address") or "").lower()
    action = infer_action(transfer, wallet_meta["address"])
    timestamp = transfer.get("block_timestamp") or transfer.get("blockTimestamp")

    return {
        "wallet_label": wallet_meta["label"],
        "wallet_address": wallet_meta["address"],
        "chain": wallet_meta["chain"],
        "action": action,
        "token_symbol": symbol,
        "token_name": transfer.get("token_name") or "",
        "token_address": token_address,
        "amount": amount,
        "timestamp": timestamp,
        "tx_hash": transfer.get("transaction_hash") or "",
        "from_address": transfer.get("from_address") or "",
        "to_address": transfer.get("to_address") or "",
    }


def enrich_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for chain, token_address in sorted(set(zip(df["chain"], df["token_address"]))):
        if not token_address:
            continue
        snapshots[(chain, token_address)] = get_token_market_snapshot(chain, token_address)
        time.sleep(0.08)

    def col(row: pd.Series, key: str):
        return snapshots.get((row["chain"], row["token_address"]), {}).get(key)

    for key in [
        "priceUsd",
        "liquidityUsd",
        "fdv",
        "marketCap",
        "volume24h",
        "priceChangeM5",
        "priceChangeH1",
        "priceChangeH6",
        "priceChangeH24",
        "pairUrl",
        "dexId",
    ]:
        df[key] = df.apply(lambda r: col(r, key), axis=1)

    return df


def build_consensus_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    buys = df[df["action"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame()

    grouped = (
        buys.groupby(["chain", "token_symbol", "token_name", "token_address"], dropna=False)
        .agg(
            wallets=("wallet_label", lambda s: ", ".join(sorted(set(s)))),
            wallet_count=("wallet_label", lambda s: len(set(s))),
            buy_events=("tx_hash", "count"),
            last_seen=("timestamp", "max"),
            price_usd=("priceUsd", "max"),
            liquidity_usd=("liquidityUsd", "max"),
            volume_24h=("volume24h", "max"),
            price_change_h24=("priceChangeH24", "max"),
            pair_url=("pairUrl", "max"),
        )
        .reset_index()
        .sort_values(["wallet_count", "buy_events", "liquidity_usd"], ascending=[False, False, False])
    )
    return grouped


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Configuration")
    st.markdown(
        "Enter wallets as one per line: `label,address,chain`  \n"
        "Example: `alpha,0x1234...,base`"
    )
    wallets_text = st.text_area(
        "Tracked wallets",
        value=(
            "alpha,0x000000000000000000000000000000000000dead,eth\n"
            "beta,0x000000000000000000000000000000000000beef,base"
        ),
        height=160,
    )
    moralis_api_key = st.text_input("Moralis API key", type="password")
    hours_back = st.slider("Lookback window (hours)", 6, 168, DEFAULT_LOOKBACK_HOURS, 6)
    limit = st.slider("Max transfers per wallet", 10, 100, DEFAULT_LIMIT, 10)
    min_wallet_consensus = st.slider("Minimum wallets buying same token", 1, 5, 2, 1)
    hide_majors = st.checkbox("Hide majors / stablecoins", value=True)
    only_buys = st.checkbox("Only show BUY signals", value=True)
    refresh = st.button("Refresh")

wallets = parse_wallets(wallets_text)

if "last_loaded" not in st.session_state:
    st.session_state.last_loaded = None
if "signals_df" not in st.session_state:
    st.session_state.signals_df = pd.DataFrame()

if refresh:
    st.cache_data.clear()

if not wallets:
    st.info("Add at least one wallet in the sidebar to begin.")
    st.stop()

if not moralis_api_key:
    st.warning("Add a Moralis API key to load live wallet transfer data.")
    st.code("pip install streamlit pandas requests")
    st.stop()

errors: List[str] = []
all_rows: List[Dict[str, Any]] = []

with st.spinner("Fetching wallet activity..."):
    for wallet in wallets:
        try:
            transfers = get_wallet_token_transfers(
                address=wallet["address"],
                chain=wallet["chain"],
                api_key=moralis_api_key,
                hours_back=hours_back,
                limit=limit,
            )
            for item in transfers:
                all_rows.append(normalize_transfer(item, wallet))
        except Exception as exc:
            errors.append(f"{wallet['label']} ({wallet['chain']}): {exc}")

signals_df = pd.DataFrame(all_rows)

if not signals_df.empty:
    signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"], errors="coerce", utc=True)
    signals_df = signals_df.sort_values("timestamp", ascending=False)
    if only_buys:
        signals_df = signals_df[signals_df["action"] == "BUY"]
    if hide_majors:
        signals_df = signals_df[~signals_df["token_symbol"].isin(IGNORE_SYMBOLS)]
    signals_df = enrich_signals(signals_df)

st.session_state.signals_df = signals_df
st.session_state.last_loaded = now_utc()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracked wallets", len(wallets))
col2.metric("Signals", len(signals_df))
col3.metric("Unique tokens", int(signals_df["token_address"].nunique()) if not signals_df.empty else 0)
col4.metric(
    "Last refresh (UTC)",
    st.session_state.last_loaded.strftime("%Y-%m-%d %H:%M:%S") if st.session_state.last_loaded else "-",
)

if errors:
    with st.expander("Wallet/API errors", expanded=False):
        for err in errors:
            st.error(err)

if signals_df.empty:
    st.info("No matching signals found for the current filters.")
    st.stop()

consensus_df = build_consensus_table(signals_df)
if not consensus_df.empty:
    consensus_df = consensus_df[consensus_df["wallet_count"] >= min_wallet_consensus]

st.subheader("Consensus buys")
if consensus_df.empty:
    st.info("No repeated buys found across the selected wallets.")
else:
    display_consensus = consensus_df.copy()
    for col in ["price_usd", "liquidity_usd", "volume_24h"]:
        display_consensus[col] = display_consensus[col].apply(fmt_num)
    display_consensus["price_change_h24"] = display_consensus["price_change_h24"].apply(
        lambda x: "-" if pd.isna(x) else f"{float(x):.2f}%"
    )
    st.dataframe(
        display_consensus[
            [
                "chain",
                "token_symbol",
                "token_name",
                "wallet_count",
                "buy_events",
                "wallets",
                "price_usd",
                "liquidity_usd",
                "volume_24h",
                "price_change_h24",
                "last_seen",
                "pair_url",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Recent wallet activity")
display = signals_df.copy()
for col in ["amount", "priceUsd", "liquidityUsd", "fdv", "marketCap", "volume24h"]:
    if col in display.columns:
        display[col] = display[col].apply(fmt_num)
for col in ["priceChangeM5", "priceChangeH1", "priceChangeH6", "priceChangeH24"]:
    if col in display.columns:
        display[col] = display[col].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.2f}%")

st.dataframe(
    display[
        [
            "timestamp",
            "wallet_label",
            "chain",
            "action",
            "token_symbol",
            "token_name",
            "amount",
            "priceUsd",
            "liquidityUsd",
            "volume24h",
            "priceChangeH24",
            "pairUrl",
            "tx_hash",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

with st.expander("How to use this app"):
    st.markdown(
        """
1. Add a Moralis API key in the sidebar.
2. Add wallets in `label,address,chain` format.
3. Click **Refresh**.
4. Start with 5-10 wallets you believe are worth tracking.
5. Watch the **Consensus buys** table first.

Heuristic used here:
- inbound ERC-20 transfer to tracked wallet = BUY
- outbound ERC-20 transfer from tracked wallet = SELL

This is intentionally simple and fast, but not perfect. Wallets can receive transfers for reasons other than buying, including transfers between owned wallets, airdrops, vesting, LP activity, or treasury moves.
        """
    )

with st.expander("Next upgrades"):
    st.markdown(
        """
- Telegram or Discord alerts for new consensus buys
- Wallet scoring based on past performance
- Exclude airdrops and suspicious low-liquidity tokens
- Add Solana support through a separate data provider
- Add paper-trading and alert backtesting
- Save tracked wallets and settings to a database
        """
    )
