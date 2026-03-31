import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

APP_TITLE = "Crypto Tracking Agent"
APP_SUBTITLE = "Track wallet swaps, score signals, and review token momentum"
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
        parts = [p.strip() for p in line.split(",")]
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
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


@st.cache_data(ttl=60)
def get_wallet_stats(address: str, chain: str, api_key: str) -> Dict[str, Any]:
    url = f"{MORALIS_BASE}/wallets/{address}/stats"
    headers = {"X-API-Key": api_key}
    params = {"chain": chain}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        return {"error": f"Moralis stats error {resp.status_code}: {resp.text[:180]}"}
    return resp.json() or {}


@st.cache_data(ttl=60)
def get_wallet_swaps(address: str, chain: str, api_key: str, hours_back: int, limit: int) -> List[Dict[str, Any]]:
    to_date = now_utc()
    from_date = to_date - timedelta(hours=hours_back)
    url = f"{MORALIS_BASE}/wallets/{address}/swaps"
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
        raise RuntimeError(f"Moralis swaps error {resp.status_code}: {resp.text[:250]}")
    data = resp.json() or {}
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
    }


def normalize_swap(item: Dict[str, Any], wallet_meta: Dict[str, str]) -> Dict[str, Any]:
    bought = item.get("bought") or {}
    sold = item.get("sold") or {}

    bought_symbol = str(bought.get("symbol") or "").upper()
    sold_symbol = str(sold.get("symbol") or "").upper()
    bought_amount = bought.get("amount") or 0
    sold_amount = sold.get("amount") or 0

    action = "BUY"
    token_symbol = bought_symbol
    token_name = bought.get("name") or ""
    token_address = str(bought.get("address") or "").lower()
    amount = bought_amount

    if bought_symbol in STABLE_SYMBOLS and sold_symbol not in STABLE_SYMBOLS and sold_symbol:
        action = "SELL"
        token_symbol = sold_symbol
        token_name = sold.get("name") or ""
        token_address = str(sold.get("address") or "").lower()
        amount = sold_amount
    elif not token_symbol and sold_symbol:
        action = "SELL"
        token_symbol = sold_symbol
        token_name = sold.get("name") or ""
        token_address = str(sold.get("address") or "").lower()
        amount = sold_amount

    return {
        "wallet_label": wallet_meta["label"],
        "wallet_address": wallet_meta["address"],
        "chain": wallet_meta["chain"],
        "action": action,
        "token_symbol": token_symbol,
        "token_name": token_name,
        "token_address": token_address,
        "amount": amount,
        "timestamp": item.get("blockTimestamp") or item.get("block_timestamp"),
        "tx_hash": item.get("transactionHash") or item.get("transaction_hash") or "",
        "sold_symbol": sold_symbol,
        "sold_amount": sold_amount,
        "bought_symbol": bought_symbol,
        "bought_amount": bought_amount,
        "pairLabel": item.get("pairLabel") or item.get("pair_label") or "",
        "exchangeName": item.get("exchangeName") or item.get("exchange_name") or "",
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


def classify_wallet_style(wallet_df: pd.DataFrame) -> str:
    if wallet_df.empty:
        return "unknown"
    total = len(wallet_df)
    unique_tokens = wallet_df["token_symbol"].nunique()
    buy_ratio = (wallet_df["action"] == "BUY").mean()
    avg_gap_minutes = 0.0
    timestamps = wallet_df["timestamp"].dropna().sort_values()
    if len(timestamps) >= 2:
        diffs = timestamps.diff().dropna().dt.total_seconds() / 60
        if not diffs.empty:
            avg_gap_minutes = float(diffs.mean())

    if total >= 8 and unique_tokens <= 3:
        return "concentrated accumulator"
    if total >= 8 and unique_tokens >= 6:
        return "rotational trader"
    if buy_ratio > 0.8 and avg_gap_minutes < 120 and total >= 5:
        return "active buyer"
    if buy_ratio < 0.35 and total >= 5:
        return "distribution / seller"
    return "mixed trader"


def build_wallet_profiles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for wallet_label, wallet_df in df.groupby("wallet_label"):
        rows.append(
            {
                "wallet_label": wallet_label,
                "chain": wallet_df["chain"].iloc[0],
                "events": len(wallet_df),
                "unique_tokens": wallet_df["token_symbol"].nunique(),
                "buy_events": int((wallet_df["action"] == "BUY").sum()),
                "sell_events": int((wallet_df["action"] == "SELL").sum()),
                "largest_buy_amount": pd.to_numeric(
                    wallet_df.loc[wallet_df["action"] == "BUY", "amount"], errors="coerce"
                ).max(),
                "style": classify_wallet_style(wallet_df),
                "last_seen": wallet_df["timestamp"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values(["events", "buy_events"], ascending=False)


def compute_signal_score(row: pd.Series) -> Tuple[int, str, List[str], List[str]]:
    score = 0
    strengths: List[str] = []
    risks: List[str] = []

    wallet_count = int(row.get("wallet_count", 0) or 0)
    buy_events = int(row.get("buy_events", 0) or 0)
    liquidity = float(row.get("liquidity_usd", 0) or 0)
    volume_24h = float(row.get("volume_24h", 0) or 0)
    price_change_h24 = float(row.get("price_change_h24", 0) or 0)

    score += min(wallet_count * 20, 40)
    if wallet_count >= 2:
        strengths.append(f"{wallet_count} wallets bought the same token")
    else:
        risks.append("Only one wallet has bought this token so far")

    score += min(buy_events * 4, 20)
    if buy_events >= 3:
        strengths.append(f"Repeated buying detected ({buy_events} buys)")

    if liquidity > 250_000:
        score += 20
        strengths.append("High liquidity")
    elif liquidity > 100_000:
        score += 15
        strengths.append("Healthy liquidity")
    elif liquidity > 50_000:
        score += 8
        strengths.append("Usable liquidity")
    else:
        score -= 18
        risks.append("Low liquidity raises slippage and exit risk")

    if volume_24h > 250_000:
        score += 15
        strengths.append("Strong 24h volume")
    elif volume_24h > 100_000:
        score += 10
        strengths.append("Solid 24h volume")
    elif volume_24h > 25_000:
        score += 5
    else:
        score -= 12
        risks.append("Thin 24h volume")

    if price_change_h24 > 25:
        score -= 15
        risks.append("Token may already be extended after a sharp 24h move")
    elif price_change_h24 > 10:
        score -= 8
        risks.append("Momentum is already elevated")
    elif -5 <= price_change_h24 <= 10:
        score += 6
        strengths.append("Price is not heavily extended")
    elif price_change_h24 < -15:
        score -= 6
        risks.append("Falling price trend could indicate weak follow-through")

    score = max(0, min(int(score), 100))

    if score >= 75:
        label = "High Conviction"
    elif score >= 60:
        label = "Strong Watch"
    elif score >= 40:
        label = "Watch"
    elif score >= 20:
        label = "Low Signal"
    else:
        label = "Ignore"

    return score, label, strengths, risks


def build_consensus_table(df: pd.DataFrame, wallet_profiles: pd.DataFrame) -> pd.DataFrame:
    buys = df[df["action"] == "BUY"].copy()
    if buys.empty:
        return pd.DataFrame()

    style_lookup = {}
    if not wallet_profiles.empty:
        style_lookup = dict(zip(wallet_profiles["wallet_label"], wallet_profiles["style"]))

    grouped = (
        buys.groupby(["chain", "token_symbol", "token_name", "token_address"], dropna=False)
        .agg(
            wallets=("wallet_label", lambda s: ", ".join(sorted(set(s)))),
            wallet_list=("wallet_label", lambda s: sorted(set(s))),
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

    grouped["wallet_styles"] = grouped["wallet_list"].apply(
        lambda wallet_list: ", ".join(sorted({style_lookup.get(w, "unknown") for w in wallet_list}))
    )

    signal_scores: List[int] = []
    signal_labels: List[str] = []
    strength_texts: List[str] = []
    risk_texts: List[str] = []

    for _, row in grouped.iterrows():
        score, label, strengths, risks = compute_signal_score(row)
        signal_scores.append(score)
        signal_labels.append(label)
        strength_texts.append("; ".join(strengths) if strengths else "-")
        risk_texts.append("; ".join(risks) if risks else "-")

    grouped["signal_score"] = signal_scores
    grouped["signal_label"] = signal_labels
    grouped["strengths"] = strength_texts
    grouped["risks"] = risk_texts

    grouped = grouped.sort_values(
        ["signal_score", "wallet_count", "buy_events", "liquidity_usd"],
        ascending=[False, False, False, False],
    )
    return grouped


def make_signal_payload(row: pd.Series) -> Dict[str, Any]:
    return {
        "token": row.get("token_symbol"),
        "token_name": row.get("token_name"),
        "chain": row.get("chain"),
        "wallet_count": int(row.get("wallet_count", 0) or 0),
        "buy_events": int(row.get("buy_events", 0) or 0),
        "wallets": [w.strip() for w in str(row.get("wallets", "")).split(",") if w.strip()],
        "wallet_styles": row.get("wallet_styles", ""),
        "price_usd": row.get("price_usd"),
        "liquidity_usd": row.get("liquidity_usd"),
        "volume_24h": row.get("volume_24h"),
        "price_change_h24": row.get("price_change_h24"),
        "signal_score": int(row.get("signal_score", 0) or 0),
        "signal_label": row.get("signal_label"),
        "strengths": row.get("strengths"),
        "risks": row.get("risks"),
        "pair_url": row.get("pair_url"),
    }


def local_ai_signal_brief(payload: Dict[str, Any]) -> Dict[str, str]:
    score = int(payload.get("signal_score", 0) or 0)
    label = str(payload.get("signal_label", "Watch"))
    token = str(payload.get("token") or "Unknown")
    wallet_count = int(payload.get("wallet_count", 0) or 0)
    buy_events = int(payload.get("buy_events", 0) or 0)
    liquidity = float(payload.get("liquidity_usd", 0) or 0)
    volume_24h = float(payload.get("volume_24h", 0) or 0)
    price_change = float(payload.get("price_change_h24", 0) or 0)
    wallet_styles = str(payload.get("wallet_styles") or "unknown")

    summary_parts = [
        f"{token} is rated {label} with a score of {score}/100.",
        f"The signal is based on {wallet_count} tracked wallet(s) and {buy_events} buy event(s).",
        f"Wallet behavior looks like: {wallet_styles}.",
    ]

    if liquidity >= 100_000:
        summary_parts.append("Liquidity is reasonably healthy for a short-term watchlist setup.")
    elif liquidity >= 50_000:
        summary_parts.append("Liquidity is usable but still thin enough to require caution.")
    else:
        summary_parts.append("Liquidity is low, so slippage and exit risk are elevated.")

    if -5 <= price_change <= 10:
        summary_parts.append("Price does not look heavily extended yet.")
    elif price_change > 10:
        summary_parts.append("Momentum is already elevated, so chasing becomes riskier.")
    else:
        summary_parts.append("Recent price weakness means follow-through is less certain.")

    if wallet_count >= 2:
        action = "Watch closely for follow-through or additional cluster buying."
    elif buy_events >= 4 and liquidity >= 50_000:
        action = "Keep it on the watchlist, but wait for another wallet to confirm the move."
    else:
        action = "Treat this as exploratory only, not a high-conviction setup."

    risk = (
        f"Key risks: {payload.get('risks', '-')}. "
        f"24h volume is {fmt_num(volume_24h)} and liquidity is {fmt_num(liquidity)}."
    )

    return {
        "summary": " ".join(summary_parts),
        "action": action,
        "risk": risk,
        "source": "local_rules",
    }


def llm_signal_brief_openai(payload: Dict[str, Any], api_key: str, model: str) -> Dict[str, str]:
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are a crypto signal analyst. Use only the JSON provided. Do not invent facts. "
        "Return strict JSON with keys: summary, action, risk. "
        "Focus on signal quality, accumulation behavior, liquidity risk, momentum risk, and whether the signal is ignore/watch/strong watch/high conviction. "
        "Keep each field under 80 words."
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": prompt},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
        text={"format": {"type": "json_object"}},
    )
    content = getattr(response, "output_text", "") or ""
    data = json.loads(content)
    return {
        "summary": str(data.get("summary", "No summary returned.")).strip(),
        "action": str(data.get("action", "No action returned.")).strip(),
        "risk": str(data.get("risk", "No risk returned.")).strip(),
        "source": f"openai:{model}",
    }


def generate_signal_brief(payload: Dict[str, Any], provider: str, api_key: str, model: str) -> Dict[str, str]:
    if provider == "OpenAI" and api_key:
        try:
            return llm_signal_brief_openai(payload, api_key, model)
        except Exception as exc:
            fallback = local_ai_signal_brief(payload)
            fallback["source"] = f"fallback_local_rules ({exc})"
            return fallback
    return local_ai_signal_brief(payload)


@st.cache_data(ttl=300)
def get_top_traders_by_token(token_address: str, chain: str, api_key: str) -> List[Dict[str, Any]]:
    url = f"{MORALIS_BASE}/erc20/{token_address}/top-gainers"
    headers = {"X-API-Key": api_key}
    params = {"chain": chain}
    resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"Moralis top traders error {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("result", []) or data.get("top_gainers", []) or data.get("topTraders", []) or []
    return []


def discover_wallet_candidates(consensus_df: pd.DataFrame, existing_wallets: List[Dict[str, str]], api_key: str, top_n_tokens: int = 3) -> pd.DataFrame:
    if consensus_df.empty:
        return pd.DataFrame()

    existing_addresses = {w["address"].lower() for w in existing_wallets}
    existing_labels = {w["label"] for w in existing_wallets}
    rows: List[Dict[str, Any]] = []
    seed_df = consensus_df.sort_values(["signal_score", "buy_events"], ascending=False).head(top_n_tokens)

    for _, token_row in seed_df.iterrows():
        token_address = str(token_row.get("token_address") or "").lower()
        chain = str(token_row.get("chain") or "eth")
        if not token_address:
            continue
        try:
            traders = get_top_traders_by_token(token_address, chain, api_key)
        except Exception:
            continue

        for idx, trader in enumerate(traders, start=1):
            address = str(
                trader.get("address")
                or trader.get("wallet_address")
                or trader.get("owner_of")
                or trader.get("wallet")
                or ""
            ).lower()
            if not address or address in existing_addresses:
                continue

            pnl = trader.get("realized_profit_usd")
            if pnl is None:
                pnl = trader.get("total_profit_usd")
            if pnl is None:
                pnl = trader.get("profit_usd")
            score = 0
            try:
                score += max(0, min(int(float(pnl or 0) / 100), 40))
            except Exception:
                pass
            score += max(0, 25 - idx)
            score += min(int(token_row.get("signal_score", 0) or 0) // 3, 25)

            label_base = str(token_row.get("token_symbol") or "seed").lower()
            auto_label = f"{label_base}_{idx}"
            while auto_label in existing_labels:
                auto_label = f"{auto_label}_x"

            rows.append({
                "suggested_label": auto_label,
                "address": address,
                "chain": chain,
                "seed_token": token_row.get("token_symbol"),
                "seed_signal_score": token_row.get("signal_score"),
                "top_trader_rank": idx,
                "profit_usd": pnl,
                "discovery_score": score,
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["discovery_score", "top_trader_rank"], ascending=[False, True])
    out = out.drop_duplicates(subset=["address", "chain"])
    return out.head(20)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("Configuration")
    st.markdown("Enter wallets as one per line: `label,address,chain`  \nExample: `alpha,0x1234...,base`")
    wallets_text = st.text_area(
        "Tracked wallets",
        value="",
        height=180,
        placeholder="alpha,0x1234...,base\nbeta,0xabcd...,eth",
    )
    default_api_key = st.secrets.get("MORALIS_API_KEY", "") if hasattr(st, "secrets") else ""
    moralis_api_key = st.text_input("Moralis API key", value=default_api_key, type="password")
    hours_back = st.slider("Lookback window (hours)", 6, 168, DEFAULT_LOOKBACK_HOURS, 6)
    limit = st.slider("Max swaps per wallet", 10, 100, DEFAULT_LIMIT, 10)
    min_wallet_consensus = st.slider("Minimum wallets buying same token", 1, 5, 2, 1)
    hide_majors = st.checkbox("Hide majors / stablecoins", value=True)
    only_buys = st.checkbox("Only show BUY signals", value=True)
    show_debug_json = st.checkbox("Show AI payload JSON", value=False)
    st.markdown("### AI analysis")
    ai_provider = st.selectbox("AI provider", ["Local rules only", "OpenAI"])
    default_openai_key = ""
    try:
        default_openai_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        default_openai_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.text_input("OpenAI API key", value=default_openai_key, type="password")
    openai_model = st.text_input("OpenAI model", value="gpt-5")
    auto_discover_wallets = st.checkbox("Auto-discover wallet candidates", value=False)
    discovery_seed_count = st.slider("Discovery seed tokens", 1, 5, 3, 1)
    refresh = st.button("Refresh")

wallets = parse_wallets(wallets_text)
if refresh:
    st.cache_data.clear()

if not wallets:
    st.info("Add at least one real wallet in the sidebar to begin.")
    st.stop()
if not moralis_api_key:
    st.warning("Add a Moralis API key to load live wallet data.")
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
raw_signals_df = pd.DataFrame(all_rows)

st.subheader("Wallet diagnostics")
if not stats_df.empty:
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
if errors:
    with st.expander("API errors", expanded=True):
        for err in errors:
            st.error(err)
if raw_signals_df.empty:
    st.warning("Moralis returned no swaps for these wallets in the current lookback window.")
    st.stop()

signals_df = raw_signals_df.copy()
signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"], errors="coerce", utc=True)
signals_df = signals_df.sort_values("timestamp", ascending=False)

if only_buys:
    signals_df = signals_df[signals_df["action"] == "BUY"]
if hide_majors:
    signals_df = signals_df[~signals_df["token_symbol"].isin(IGNORE_SYMBOLS)]

signals_df = enrich_signals(signals_df)
wallet_profiles_df = build_wallet_profiles(signals_df)
consensus_df = build_consensus_table(signals_df, wallet_profiles_df)
if not consensus_df.empty:
    consensus_df = consensus_df[consensus_df["wallet_count"] >= min_wallet_consensus]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tracked wallets", len(wallets))
col2.metric("Signals", len(signals_df))
col3.metric("Unique tokens", int(signals_df["token_address"].nunique()) if not signals_df.empty else 0)
col4.metric("Last refresh (UTC)", now_utc().strftime("%Y-%m-%d %H:%M:%S"))

if signals_df.empty:
    st.info("Activity exists, but nothing passed the current filters. Turn off 'Hide majors / stablecoins' or 'Only show BUY signals'.")
    st.stop()

st.subheader("Wallet intelligence")
if wallet_profiles_df.empty:
    st.info("No wallet profiles available yet.")
else:
    wallet_profiles_display = wallet_profiles_df.copy()
    wallet_profiles_display["largest_buy_amount"] = wallet_profiles_display["largest_buy_amount"].apply(fmt_num)
    st.dataframe(wallet_profiles_display, use_container_width=True, hide_index=True)

st.subheader("Consensus buys + signal intelligence")
if consensus_df.empty:
    st.info("No repeated buys found across the selected wallets.")
else:
    display_consensus = consensus_df.copy()
    for col in ["price_usd", "liquidity_usd", "volume_24h"]:
        display_consensus[col] = display_consensus[col].apply(fmt_num)
    display_consensus["price_change_h24"] = display_consensus["price_change_h24"].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.2f}%")
    st.dataframe(
        display_consensus[[
            "signal_score",
            "signal_label",
            "chain",
            "token_symbol",
            "token_name",
            "wallet_count",
            "buy_events",
            "wallets",
            "wallet_styles",
            "price_usd",
            "liquidity_usd",
            "volume_24h",
            "price_change_h24",
            "strengths",
            "risks",
            "last_seen",
            "pair_url",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    signal_options = [f"{row.token_symbol} | {row.chain} | {row.signal_label} | {row.signal_score}" for _, row in consensus_df.iterrows()]
    selected_option = st.selectbox("Analyze one signal", signal_options)
    selected_index = signal_options.index(selected_option)
    selected_row = consensus_df.iloc[selected_index]
    payload = make_signal_payload(selected_row)
    provider_name = "OpenAI" if ai_provider == "OpenAI" else "Local rules only"
    analysis = generate_signal_brief(payload, provider_name, openai_api_key, openai_model)

    st.markdown("### AI signal brief")
    st.caption(f"Analysis source: {analysis.get('source', 'unknown')}")
    st.write(analysis["summary"])
    st.write(f"**Suggested action:** {analysis['action']}")
    st.write(f"**Risk note:** {analysis['risk']}")

    if show_debug_json:
        st.code(json.dumps(payload, indent=2, default=str), language="json")

if auto_discover_wallets:
    st.subheader("Auto-discovered wallet candidates")
    candidates_df = discover_wallet_candidates(consensus_df, wallets, moralis_api_key, discovery_seed_count)
    if candidates_df.empty:
        st.info("No wallet candidates were discovered from the current top signals.")
    else:
        candidates_display = candidates_df.copy()
        candidates_display["profit_usd"] = candidates_display["profit_usd"].apply(fmt_num)
        st.dataframe(candidates_display, use_container_width=True, hide_index=True)

        # Fix string construction for suggested wallets
        lines = []
        for row in candidates_df.itertuples(index=False):
            lines.append(f"{row.suggested_label},{row.address},{row.chain}")
        suggested_wallets_text = "
".join(lines)
        st.markdown("### Suggested wallets to add")
        st.code(suggested_wallets_text, language="text")

st.subheader("Recent wallet swaps")
display = signals_df.copy()
for col in ["amount", "sold_amount", "bought_amount", "priceUsd", "liquidityUsd", "fdv", "marketCap", "volume24h"]:
    if col in display.columns:
        display[col] = display[col].apply(fmt_num)
for col in ["priceChangeM5", "priceChangeH1", "priceChangeH6", "priceChangeH24"]:
    if col in display.columns:
        display[col] = display[col].apply(lambda x: "-" if pd.isna(x) else f"{float(x):.2f}%")

st.dataframe(
    display[[
        "timestamp",
        "wallet_label",
        "chain",
        "action",
        "token_symbol",
        "token_name",
        "amount",
        "sold_symbol",
        "sold_amount",
        "bought_symbol",
        "bought_amount",
        "exchangeName",
        "priceUsd",
        "liquidityUsd",
        "volume24h",
        "priceChangeH24",
        "pairUrl",
        "tx_hash",
    ]],
    use_container_width=True,
    hide_index=True,
)

with st.expander("How to use this app"):
    st.markdown(
        """
1. Add your Moralis API key.
2. Add real wallets in `label,address,chain` format.
3. Click **Refresh**.
4. Check **Wallet intelligence** to see whether a wallet looks rotational, concentrated, or mixed.
5. Check **Consensus buys + signal intelligence** to find the highest-scoring setups.
6. Read the **AI signal brief** for a plain-English interpretation.

This version adds a built-in signal intelligence layer using deterministic scoring plus either a local AI-style explanation module or a real OpenAI model through the Responses API. It can also auto-discover wallet candidates from the strongest current token signals by querying token top traders.
        """
    )
