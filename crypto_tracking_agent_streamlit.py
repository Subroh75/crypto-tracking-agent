import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st

APP_TITLE = "Crypto Tracking Agent"
APP_SUBTITLE = "Two-panel scanner: on-chain whales + social narrative fusion"
MORALIS_BASE = "https://deep-index.moralis.io/api/v2.2"
REQUEST_TIMEOUT = 20
WALLETS_FILE = "wallets.txt"
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


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_num(value: Any) -> str:
    try:
        num = float(value)
    except Exception:
        return "-"
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"{num / 1_000:.2f}K"
    if abs(num) >= 1:
        return f"{num:.2f}"
    return f"{num:.6f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def load_wallets_from_file() -> str:
    try:
        with open(WALLETS_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def save_wallets_to_file(wallets_text: str) -> Tuple[bool, str]:
    try:
        with open(WALLETS_FILE, "w", encoding="utf-8") as f:
            f.write(wallets_text.strip() + ("\n" if wallets_text.strip() else ""))
        return True, "Wallet list saved"
    except Exception as exc:
        return False, f"Could not save wallets: {exc}"


def default_wallets_text() -> str:
    saved = load_wallets_from_file()
    if saved:
        return saved
    return "\n".join([
        "alpha1,0x1dc89ab25ab5d8714fcf9ee4bd9c9a58debeb4d8,eth",
        "alpha2,0xc5e9f816994d3eb91b556bc8d0a0cbe44a674909,eth",
        "alpha3,0x31c28fe6dbc15930d4d670af8f1d7f4ee4a6cd95,eth",
        "alpha4,0x31a0a6ce4a67dcb2ff37b1b3e0cbf32a599f6d5a,eth",
        "alpha5,0xb7b78a8a908acf3c72a9c30c4e0a413c6b020611,eth",
    ])


def parse_wallets(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        label = parts[0]
        address = parts[1].lower()
        chain = parts[2].lower() if len(parts) >= 3 else "eth"
        if not address.startswith("0x"):
            continue
        if chain not in SUPPORTED_CHAINS:
            chain = "eth"
        key = (address, chain)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"label": label, "address": address, "chain": chain})
    return rows


def parse_social_watchlist(text: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        token_symbol = parts[0].upper()
        mentions = int(float(parts[1])) if len(parts) >= 2 and parts[1] else 1
        influencers = int(float(parts[2])) if len(parts) >= 3 and parts[2] else 1
        sentiment = safe_float(parts[3], 0.0) if len(parts) >= 4 else 0.0
        note = parts[4] if len(parts) >= 5 else ""
        rows.append({
            "token_symbol": token_symbol,
            "mentions": mentions,
            "influencers": influencers,
            "sentiment": sentiment,
            "note": note,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=120)
def get_wallet_stats(address: str, chain: str, api_key: str) -> Dict[str, Any]:
    url = f"{MORALIS_BASE}/wallets/{address}/stats"
    resp = requests.get(url, headers={"X-API-Key": api_key}, params={"chain": chain}, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        return {"error": f"{resp.status_code}: {resp.text[:120]}"}
    return resp.json() or {}


@st.cache_data(ttl=120)
def get_wallet_swaps(address: str, chain: str, api_key: str, hours_back: int, limit: int) -> List[Dict[str, Any]]:
    to_date = now_utc()
    from_date = to_date - timedelta(hours=hours_back)
    url = f"{MORALIS_BASE}/wallets/{address}/swaps"
    params = {
        "chain": chain,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "limit": max(1, min(limit, 100)),
        "order": "DESC",
    }
    resp = requests.get(url, headers={"X-API-Key": api_key}, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:140]}")
    data = resp.json() or {}
    if isinstance(data, list):
        return data
    return data.get("result", []) or []


@st.cache_data(ttl=180)
def get_token_market_snapshot(chain: str, token_address: str) -> Dict[str, Any]:
    if not token_address:
        return {}
    ds_chain = DEXSCREENER_CHAIN_MAP.get(chain, chain)
    url = f"https://api.dexscreener.com/token-pairs/v1/{ds_chain}/{token_address}"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception:
        return {}
    if resp.status_code != 200:
        return {}
    pairs = resp.json() or []
    if not isinstance(pairs, list) or not pairs:
        return {}

    def rank_pair(pair: Dict[str, Any]) -> Tuple[float, float]:
        return (
            safe_float(safe_get(pair, "liquidity", "usd", default=0)),
            safe_float(safe_get(pair, "volume", "h24", default=0)),
        )

    best = sorted(pairs, key=rank_pair, reverse=True)[0]
    return {
        "priceUsd": safe_get(best, "priceUsd"),
        "liquidityUsd": safe_get(best, "liquidity", "usd"),
        "volume24h": safe_get(best, "volume", "h24"),
        "priceChangeH24": safe_get(best, "priceChange", "h24"),
        "pairUrl": best.get("url"),
    }


def normalize_swap(item: Dict[str, Any], wallet_meta: Dict[str, str]) -> Dict[str, Any]:
    bought = item.get("bought") or {}
    sold = item.get("sold") or {}
    bought_symbol = str(bought.get("symbol") or "").upper()
    sold_symbol = str(sold.get("symbol") or "").upper()
    action = "BUY"
    token_symbol = bought_symbol
    token_name = str(bought.get("name") or "")
    token_address = str(bought.get("address") or "").lower()
    amount = bought.get("amount") or 0

    if bought_symbol in STABLE_SYMBOLS and sold_symbol and sold_symbol not in STABLE_SYMBOLS:
        action = "SELL"
        token_symbol = sold_symbol
        token_name = str(sold.get("name") or "")
        token_address = str(sold.get("address") or "").lower()
        amount = sold.get("amount") or 0

    return {
        "timestamp": item.get("blockTimestamp") or item.get("block_timestamp"),
        "wallet_label": wallet_meta["label"],
        "wallet_address": wallet_meta["address"],
        "chain": wallet_meta["chain"],
        "action": action,
        "token_symbol": token_symbol,
        "token_name": token_name,
        "token_address": token_address,
        "amount": amount,
        "sold_symbol": sold_symbol,
        "sold_amount": sold.get("amount") or 0,
        "bought_symbol": bought_symbol,
        "bought_amount": bought.get("amount") or 0,
        "exchangeName": item.get("exchangeName") or item.get("exchange_name") or "",
        "tx_hash": item.get("transactionHash") or item.get("transaction_hash") or "",
    }


def enrich_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    snapshots: Dict[Tuple[str, str], Dict[str, Any]] = {}
    unique_pairs = set(zip(df["chain"], df["token_address"]))
    for chain, token_address in unique_pairs:
        snapshots[(chain, token_address)] = get_token_market_snapshot(chain, token_address)
    for key in ["priceUsd", "liquidityUsd", "volume24h", "priceChangeH24", "pairUrl"]:
        df[key] = df.apply(lambda r: snapshots.get((r["chain"], r["token_address"]), {}).get(key), axis=1)
    return df


def classify_wallet_style(wallet_df: pd.DataFrame) -> str:
    total = len(wallet_df)
    unique_tokens = wallet_df["token_symbol"].nunique()
    buy_ratio = float((wallet_df["action"] == "BUY").mean()) if total else 0
    if total >= 8 and unique_tokens <= 3:
        return "concentrated accumulator"
    if total >= 8 and unique_tokens >= 6:
        return "rotational trader"
    if buy_ratio >= 0.8 and total >= 5:
        return "active buyer"
    if buy_ratio <= 0.35 and total >= 5:
        return "distribution / seller"
    return "mixed trader"


def build_wallet_profiles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for wallet_label, wallet_df in df.groupby("wallet_label"):
        rows.append({
            "wallet_label": wallet_label,
            "chain": wallet_df["chain"].iloc[0],
            "events": len(wallet_df),
            "unique_tokens": wallet_df["token_symbol"].nunique(),
            "buy_events": int((wallet_df["action"] == "BUY").sum()),
            "sell_events": int((wallet_df["action"] == "SELL").sum()),
            "largest_buy_amount": pd.to_numeric(wallet_df.loc[wallet_df["action"] == "BUY", "amount"], errors="coerce").max(),
            "style": classify_wallet_style(wallet_df),
            "last_seen": wallet_df["timestamp"].max(),
        })
    return pd.DataFrame(rows).sort_values(["events", "buy_events"], ascending=False)


def score_signal(wallet_count: int, buy_events: int, liquidity: float, volume_24h: float, price_change_h24: float) -> Tuple[int, str]:
    score = 0
    score += min(wallet_count * 20, 40)
    score += min(buy_events * 4, 20)
    if liquidity > 250_000:
        score += 20
    elif liquidity > 100_000:
        score += 15
    elif liquidity > 50_000:
        score += 8
    else:
        score -= 18
    if volume_24h > 250_000:
        score += 15
    elif volume_24h > 100_000:
        score += 10
    elif volume_24h > 25_000:
        score += 5
    else:
        score -= 12
    if price_change_h24 > 25:
        score -= 15
    elif price_change_h24 > 10:
        score -= 8
    elif -5 <= price_change_h24 <= 10:
        score += 6
    elif price_change_h24 < -15:
        score -= 6
    score = max(0, min(int(score), 100))
    if score >= 75:
        return score, "High Conviction"
    if score >= 60:
        return score, "Strong Watch"
    if score >= 40:
        return score, "Watch"
    if score >= 20:
        return score, "Low Signal"
    return score, "Ignore"


def classify_setup(price_change_h24: float, wallet_count: int, buy_events: int) -> str:
    if wallet_count >= 2 and price_change_h24 < 10 and buy_events >= 2:
        return "Early Accumulation"
    if wallet_count >= 2 and 10 <= price_change_h24 <= 25:
        return "Momentum Ignition"
    if price_change_h24 > 25:
        return "Extended"
    if wallet_count == 1 and buy_events >= 3 and price_change_h24 < 12:
        return "Single-Wallet Accumulation"
    return "Developing"


def classify_entry_signal(setup_type: str) -> str:
    if setup_type == "Early Accumulation":
        return "Stalk entry / small starter"
    if setup_type == "Momentum Ignition":
        return "Enter on continuation"
    if setup_type == "Extended":
        return "Wait for pullback"
    if setup_type == "Single-Wallet Accumulation":
        return "Wait for second wallet confirmation"
    return "Monitor only"


def classify_risk(liquidity: float, wallet_count: int) -> str:
    if liquidity > 250_000 and wallet_count >= 2:
        return "Low"
    if liquidity > 100_000:
        return "Medium"
    return "High"


def build_consensus_table(df: pd.DataFrame, wallet_profiles: pd.DataFrame) -> pd.DataFrame:
    buys = df[df["action"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame()
    style_lookup = dict(zip(wallet_profiles["wallet_label"], wallet_profiles["style"])) if not wallet_profiles.empty else {}
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
    )
    grouped["wallet_styles"] = grouped["wallets"].apply(lambda text: ", ".join(sorted({style_lookup.get(w.strip(), "unknown") for w in text.split(",") if w.strip()})))
    grouped["signal_score"] = grouped.apply(lambda r: score_signal(int(r["wallet_count"]), int(r["buy_events"]), safe_float(r["liquidity_usd"]), safe_float(r["volume_24h"]), safe_float(r["price_change_h24"]))[0], axis=1)
    grouped["signal_label"] = grouped.apply(lambda r: score_signal(int(r["wallet_count"]), int(r["buy_events"]), safe_float(r["liquidity_usd"]), safe_float(r["volume_24h"]), safe_float(r["price_change_h24"]))[1], axis=1)
    grouped["setup_type"] = grouped.apply(lambda r: classify_setup(safe_float(r["price_change_h24"]), int(r["wallet_count"]), int(r["buy_events"])), axis=1)
    grouped["entry_signal"] = grouped["setup_type"].apply(classify_entry_signal)
    grouped["risk_level"] = grouped.apply(lambda r: classify_risk(safe_float(r["liquidity_usd"]), int(r["wallet_count"])), axis=1)
    return grouped.sort_values(["signal_score", "wallet_count", "buy_events"], ascending=False)


def build_social_panel(social_df: pd.DataFrame) -> pd.DataFrame:
    if social_df.empty:
        return pd.DataFrame()
    out = social_df.copy()
    out["social_score"] = out.apply(lambda r: min(100, int(r["mentions"]) * 2 + int(r["influencers"]) * 6 + (20 if safe_float(r["sentiment"]) > 0.6 else 10 if safe_float(r["sentiment"]) > 0.2 else -10 if safe_float(r["sentiment"]) < -0.2 else 0)), axis=1)
    out["social_score"] = out["social_score"].clip(lower=0)
    return out.sort_values(["social_score", "mentions", "influencers"], ascending=False)


def build_fusion_panel(consensus_df: pd.DataFrame, social_df: pd.DataFrame) -> pd.DataFrame:
    if consensus_df.empty and social_df.empty:
        return pd.DataFrame()
    if consensus_df.empty:
        out = social_df.copy()
        out["chain"] = "eth"
        out["signal_score"] = 0
        out["wallet_count"] = 0
        out["buy_events"] = 0
        out["wallets"] = ""
        out["wallet_styles"] = ""
        out["liquidity_usd"] = None
        out["volume_24h"] = None
        out["price_change_h24"] = None
        out["setup_type"] = "No on-chain"
        out["entry_signal"] = "Monitor only"
        out["risk_level"] = "Unknown"
        out["fusion_score"] = out["social_score"]
        out["fusion_label"] = out["fusion_score"].apply(lambda s: "Narrative only" if s >= 20 else "Ignore")
        return out
    if social_df.empty:
        out = consensus_df.copy()
        out["mentions"] = 0
        out["influencers"] = 0
        out["sentiment"] = 0.0
        out["note"] = ""
        out["social_score"] = 0
        out["fusion_score"] = out["signal_score"]
        out["fusion_label"] = out["fusion_score"].apply(lambda s: "Whales only" if s >= 20 else "Ignore")
        return out
    merged = consensus_df.merge(social_df, on="token_symbol", how="outer")
    for col in ["signal_score", "social_score", "wallet_count", "buy_events", "mentions", "influencers"]:
        merged[col] = merged[col].fillna(0)
    for col in ["wallets", "wallet_styles", "note", "setup_type", "entry_signal", "risk_level"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")
    merged["chain"] = merged["chain"].fillna("eth")
    merged["fusion_score"] = (merged["signal_score"] * 0.65 + merged["social_score"] * 0.35).round().astype(int)

    def label_row(row: pd.Series) -> str:
        on_chain = int(row.get("signal_score", 0) or 0)
        social = int(row.get("social_score", 0) or 0)
        fusion = int(row.get("fusion_score", 0) or 0)
        if on_chain >= 40 and social >= 35:
            return "Narrative + whales"
        if on_chain >= 40:
            return "Whales only"
        if social >= 35:
            return "Narrative only"
        if fusion >= 25:
            return "Early watch"
        return "Ignore"

    merged["fusion_label"] = merged.apply(label_row, axis=1)
    return merged.sort_values(["fusion_score", "signal_score", "social_score"], ascending=False)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

if "wallets_text" not in st.session_state:
    st.session_state.wallets_text = default_wallets_text()

with st.sidebar:
    st.header("Configuration")
    wallets_text = st.text_area("Tracked wallets", value=st.session_state.wallets_text, height=220)
    st.session_state.wallets_text = wallets_text

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save wallets"):
            ok, msg = save_wallets_to_file(wallets_text)
            st.success(msg) if ok else st.error(msg)
    with c2:
        if st.button("Reload file"):
            st.session_state.wallets_text = default_wallets_text()
            st.rerun()

    try:
        default_moralis_key = st.secrets.get("MORALIS_API_KEY", "")
    except Exception:
        default_moralis_key = os.getenv("MORALIS_API_KEY", "")
    moralis_api_key = st.text_input("Moralis API key", value=default_moralis_key, type="password")
    hours_back = st.slider("Lookback window (hours)", 6, 168, 72, 6)
    limit = st.slider("Max swaps per wallet", 10, 100, 50, 10)
    min_wallet_consensus = st.slider("Minimum wallets buying same token", 1, 5, 1, 1)
    hide_majors = st.checkbox("Hide majors / stablecoins", value=False)
    only_buys = st.checkbox("Only show BUY signals", value=False)

    st.markdown("### Social watchlist")
    social_watchlist_text = st.text_area(
        "Narrative inputs",
        value="YGG,18,4,0.72,gaming accounts\nAIXBT,12,3,0.55,AI trading narrative\nSABAI,8,2,0.31,small-cap discussion",
        height=140,
        help="Format: ticker,mentions,influencers,sentiment,note",
    )

    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

wallets = parse_wallets(st.session_state.wallets_text)
if not wallets:
    st.info("Add at least one valid wallet in label,address,chain format.")
    st.stop()
if not moralis_api_key:
    st.warning("Add your Moralis API key to continue.")
    st.stop()

stats_rows: List[Dict[str, Any]] = []
errors: List[str] = []
all_rows: List[Dict[str, Any]] = []

with st.spinner("Fetching wallet activity..."):
    for wallet in wallets:
        stats = get_wallet_stats(wallet["address"], wallet["chain"], moralis_api_key)
        stats_rows.append({
            "wallet_label": wallet["label"],
            "chain": wallet["chain"],
            "address": wallet["address"],
            "token_transfers_total": safe_get(stats, "token_transfers", "total", default=0),
            "transactions_total": safe_get(stats, "transactions", "total", default=0),
            "stats_error": stats.get("error", "") if isinstance(stats, dict) else "",
        })
        try:
            swaps = get_wallet_swaps(wallet["address"], wallet["chain"], moralis_api_key, hours_back, limit)
            for item in swaps:
                all_rows.append(normalize_swap(item, wallet))
        except Exception as exc:
            errors.append(f"{wallet['label']} ({wallet['chain']}): {exc}")

stats_df = pd.DataFrame(stats_rows)
raw_df = pd.DataFrame(all_rows)
social_df = build_social_panel(parse_social_watchlist(social_watchlist_text))

st.subheader("Wallet diagnostics")
if not stats_df.empty:
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
if errors:
    for err in errors:
        st.error(err)
if raw_df.empty:
    st.warning("No swaps were returned for the current wallets and time window.")
    st.stop()

raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce", utc=True)
raw_df = raw_df.sort_values("timestamp", ascending=False)

signals_df = raw_df.copy()
if only_buys:
    signals_df = signals_df[signals_df["action"] == "BUY"]
if hide_majors:
    signals_df = signals_df[~signals_df["token_symbol"].isin(IGNORE_SYMBOLS)]
signals_df = enrich_signals(signals_df)
if signals_df.empty:
    st.info("Data was found, but nothing passed the current filters.")
    st.stop()

wallet_profiles_df = build_wallet_profiles(signals_df)
consensus_df = build_consensus_table(signals_df, wallet_profiles_df)
if not consensus_df.empty:
    consensus_df = consensus_df[consensus_df["wallet_count"] >= min_wallet_consensus]
fusion_df = build_fusion_panel(consensus_df, social_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tracked wallets", len(wallets))
m2.metric("Signals", len(signals_df))
m3.metric("Unique tokens", int(signals_df["token_address"].nunique()))
m4.metric("Last refresh (UTC)", now_utc().strftime("%Y-%m-%d %H:%M:%S"))

st.subheader("Wallet intelligence")
if wallet_profiles_df.empty:
    st.info("No wallet profiles available.")
else:
    display_profiles = wallet_profiles_df.copy()
    display_profiles["largest_buy_amount"] = display_profiles["largest_buy_amount"].apply(fmt_num)
    st.dataframe(display_profiles, use_container_width=True, hide_index=True)

left_col, right_col = st.columns(2)
with left_col:
    st.subheader("On-chain signals")
    if consensus_df.empty:
        st.info("No consensus tokens found yet.")
    else:
        display_consensus = consensus_df.copy()
        for col in ["price_usd", "liquidity_usd", "volume_24h"]:
            display_consensus[col] = display_consensus[col].apply(fmt_num)
        display_consensus["price_change_h24"] = display_consensus["price_change_h24"].apply(lambda x: "-" if pd.isna(x) else f"{safe_float(x):.2f}%")
        st.dataframe(display_consensus[[
            "signal_score", "signal_label", "setup_type", "entry_signal", "risk_level", "chain",
            "token_symbol", "token_name", "wallet_count", "buy_events", "wallets", "wallet_styles",
            "price_usd", "liquidity_usd", "volume_24h", "price_change_h24", "last_seen", "pair_url"
        ]], use_container_width=True, hide_index=True)

with right_col:
    st.subheader("Social buzz")
    if social_df.empty:
        st.info("No social watchlist rows provided.")
    else:
        social_display = social_df.copy()
        social_display["sentiment"] = social_display["sentiment"].apply(lambda x: f"{safe_float(x):.2f}")
        st.dataframe(social_display, use_container_width=True, hide_index=True)

st.subheader("Fusion scanner")
if fusion_df.empty:
    st.info("No fusion signals yet.")
else:
    fusion_display = fusion_df.copy()
    for col in ["liquidity_usd", "volume_24h"]:
        if col in fusion_display.columns:
            fusion_display[col] = fusion_display[col].apply(fmt_num)
    if "price_change_h24" in fusion_display.columns:
        fusion_display["price_change_h24"] = fusion_display["price_change_h24"].apply(lambda x: "-" if pd.isna(x) else f"{safe_float(x):.2f}%")
    show_cols = [c for c in [
        "fusion_score", "fusion_label", "token_symbol", "chain", "signal_score", "social_score",
        "setup_type", "entry_signal", "risk_level", "wallet_count", "buy_events", "mentions",
        "influencers", "sentiment", "wallets", "wallet_styles", "liquidity_usd", "volume_24h",
        "price_change_h24", "note", "pair_url"
    ] if c in fusion_display.columns]
    st.dataframe(fusion_display[show_cols], use_container_width=True, hide_index=True)

    options = [f"{row.token_symbol} | {row.fusion_label} | {row.fusion_score}" for _, row in fusion_df.iterrows()]
    selected = st.selectbox("Analyze one fusion signal", options)
    selected_row = fusion_df.iloc[options.index(selected)]
    st.write(
        f"{selected_row.get('token_symbol', 'Unknown')} is classified as {selected_row.get('fusion_label', 'Watch')} with fusion score {int(selected_row.get('fusion_score', 0) or 0)}. "
        f"Setup: {selected_row.get('setup_type', 'Unknown')}. Entry: {selected_row.get('entry_signal', 'Monitor only')}. Risk: {selected_row.get('risk_level', 'Unknown')}. "
        f"On-chain score {int(selected_row.get('signal_score', 0) or 0)}, social score {int(selected_row.get('social_score', 0) or 0)}. "
        f"Wallets: {selected_row.get('wallets', '') or 'none'}. Social note: {selected_row.get('note', '') or 'none'}."
    )

st.subheader("Recent wallet swaps")
display = signals_df.copy()
for col in ["amount", "sold_amount", "bought_amount", "priceUsd", "liquidityUsd", "volume24h"]:
    if col in display.columns:
        display[col] = display[col].apply(fmt_num)
if "priceChangeH24" in display.columns:
    display["priceChangeH24"] = display["priceChangeH24"].apply(lambda x: "-" if pd.isna(x) else f"{safe_float(x):.2f}%")

st.dataframe(display[[
    "timestamp", "wallet_label", "chain", "action", "token_symbol", "token_name", "amount",
    "sold_symbol", "sold_amount", "bought_symbol", "bought_amount", "exchangeName", "priceUsd",
    "liquidityUsd", "volume24h", "priceChangeH24", "pairUrl", "tx_hash"
]], use_container_width=True, hide_index=True)

with st.expander("Notes"):
    st.markdown(
        """
This version includes entry intelligence.

New fields:
- `setup_type`
- `entry_signal`
- `risk_level`

Interpretation:
- Early Accumulation = stalk entry / small starter
- Momentum Ignition = enter on continuation
- Extended = wait for pullback
- Single-Wallet Accumulation = wait for second wallet confirmation
        """
    )
