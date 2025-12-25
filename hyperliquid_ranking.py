"""
Hyperliquid Token Relative Strength Ranking System
With Telegram Notifications for Changes
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Tuple, Optional
import time

# ===================== CONFIGURATION =====================
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
VOLUME_THRESHOLD = 300000  # $300k 24h volume
RSI_PERIOD = 14
MA_PERIOD = 14
SIGNAL_LINE = 50
LOOKBACK_DAYS = 60  # Need enough data for RSI+MA calculation
RESULTS_FILE = "top6_history.json"
LAST_TOP6_FILE = "last_top6.json"
MAX_HISTORY_DAYS = 30

# Telegram configuration (loaded from environment variables)
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# ===================== API FUNCTIONS =====================

def fetch_perpetuals_meta() -> List[str]:
    """Fetch all available perpetual tokens from Hyperliquid"""
    payload = {"type": "meta"}
    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        tokens = [asset['name'] for asset in data['universe']]
        print(f"✅ Fetched {len(tokens)} perpetual tokens")
        return tokens
    except Exception as e:
        print(f"❌ Error fetching perpetuals meta: {e}")
        return []

def fetch_asset_contexts() -> Dict:
    """Fetch 24h volume and other metrics for all tokens"""
    payload = {"type": "metaAndAssetCtxs"}
    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse volume data
        volume_data = {}
        for i, ctx in enumerate(data[1]):
            token_name = data[0]['universe'][i]['name']
            volume_24h = float(ctx.get('dayNtlVlm', 0))  # 24h notional volume
            volume_data[token_name] = volume_24h
        
        print(f"✅ Fetched volume data for {len(volume_data)} tokens")
        return volume_data
    except Exception as e:
        print(f"❌ Error fetching asset contexts: {e}")
        return {}

def fetch_daily_candles(token: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch daily OHLCV data for a token"""
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)
    
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": token,
            "interval": "1d",
            "startTime": start_time,
            "endTime": end_time
        }
    }
    
    try:
        response = requests.post(HYPERLIQUID_API, json=payload, timeout=10)
        response.raise_for_status()
        candles = response.json()
        
        if not candles:
            return pd.DataFrame()
        
        # Parse candle data
        df = pd.DataFrame(candles)
        df['close'] = df['c'].astype(float)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.sort_values('timestamp')
        
        return df[['timestamp', 'close']]
    except Exception as e:
        print(f"⚠️  Error fetching candles for {token}: {e}")
        return pd.DataFrame()

# ===================== TECHNICAL INDICATORS =====================

def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi_ma(rsi: pd.Series, period: int = MA_PERIOD) -> float:
    """Calculate moving average of RSI and return latest value"""
    rsi_ma = rsi.rolling(window=period).mean()
    return rsi_ma.iloc[-1] if not rsi_ma.empty else 50.0

# ===================== RATIO ANALYSIS =====================

def create_ratio_series(token1_data: pd.DataFrame, token2_data: pd.DataFrame) -> pd.Series:
    """Create ratio series from two token price series"""
    # Merge on timestamp
    merged = pd.merge(token1_data, token2_data, on='timestamp', suffixes=('_1', '_2'))
    ratio = merged['close_1'] / merged['close_2']
    return ratio

def analyze_ratio(token1: str, token2: str, data_cache: Dict) -> Tuple[str, str, float]:
    """
    Analyze a ratio pair and return winner, loser, and RSI-MA value
    Returns: (winner_token, loser_token, rsi_ma_value)
    """
    if token1 not in data_cache or token2 not in data_cache:
        return None, None, 50.0
    
    token1_data = data_cache[token1]
    token2_data = data_cache[token2]
    
    if token1_data.empty or token2_data.empty:
        return None, None, 50.0
    
    # Create ratio: token1/token2
    ratio_series = create_ratio_series(token1_data, token2_data)
    
    if len(ratio_series) < RSI_PERIOD + MA_PERIOD:
        return None, None, 50.0
    
    # Calculate RSI
    rsi = calculate_rsi(ratio_series, RSI_PERIOD)
    
    # Calculate RSI-MA
    rsi_ma = calculate_rsi_ma(rsi, MA_PERIOD)
    
    # Determine winner/loser based on RSI-MA vs signal line (50)
    if rsi_ma > SIGNAL_LINE:
        return token1, token2, rsi_ma  # token1 wins
    else:
        return token2, token1, rsi_ma  # token2 wins

# ===================== SCORING ENGINE =====================

def calculate_scores(tokens: List[str], data_cache: Dict) -> Dict[str, int]:
    """
    Calculate scores for all tokens based on ratio analysis
    Each win = +1, each loss = 0
    """
    scores = {token: 0 for token in tokens}
    total_comparisons = 0
    successful_comparisons = 0
    
    print(f"\n🔄 Analyzing {len(tokens) * (len(tokens) - 1)} ratio pairs...")
    
    # Analyze all possible pairs (including both directions)
    for i, token1 in enumerate(tokens):
        for token2 in tokens:
            if token1 == token2:
                continue
            
            total_comparisons += 1
            winner, loser, rsi_ma = analyze_ratio(token1, token2, data_cache)
            
            if winner and loser:
                scores[winner] += 1
                successful_comparisons += 1
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1}/{len(tokens)} tokens...")
    
    print(f"✅ Completed {successful_comparisons}/{total_comparisons} ratio analyses")
    return scores

# ===================== TELEGRAM NOTIFICATIONS =====================

def send_telegram_message(message: str) -> bool:
    """Send a message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram not configured (missing bot token or chat ID)")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ Telegram notification sent successfully")
        return True
    except Exception as e:
        print(f"❌ Error sending Telegram notification: {e}")
        return False

def load_last_top6() -> Optional[Dict]:
    """Load the previous Top 6 results for comparison"""
    if os.path.exists(LAST_TOP6_FILE):
        try:
            with open(LAST_TOP6_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading last Top 6: {e}")
    return None

def save_last_top6(top6: List[Tuple[str, int]]):
    """Save current Top 6 for next comparison"""
    data = {
        "tokens": [token for token, score in top6],
        "scores": [score for token, score in top6],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        with open(LAST_TOP6_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved current Top 6 to {LAST_TOP6_FILE}")
    except Exception as e:
        print(f"❌ Error saving last Top 6: {e}")

def detect_changes(previous: Optional[Dict], current: List[Tuple[str, int]]) -> Optional[Dict]:
    """
    Detect changes between previous and current Top 6
    Returns a dict with change details or None if no changes
    """
    if not previous:
        return None  # First run, no comparison possible
    
    prev_tokens = previous.get('tokens', [])
    prev_scores = previous.get('scores', [])
    
    curr_tokens = [token for token, score in current]
    curr_scores = [score for token, score in current]
    
    # Check if there are any changes
    if prev_tokens == curr_tokens and prev_scores == curr_scores:
        return None  # No changes
    
    changes = {
        "new_entries": [],
        "dropped_out": [],
        "position_changes": [],
        "score_changes": [],
        "unchanged": []
    }
    
    # Detect new entries
    for i, token in enumerate(curr_tokens):
        if token not in prev_tokens:
            changes["new_entries"].append({
                "token": token,
                "position": i + 1,
                "score": curr_scores[i]
            })
    
    # Detect dropped tokens
    for i, token in enumerate(prev_tokens):
        if token not in curr_tokens:
            changes["dropped_out"].append({
                "token": token,
                "position": i + 1,
                "score": prev_scores[i]
            })
    
    # Detect position and score changes
    for i, token in enumerate(curr_tokens):
        if token in prev_tokens:
            prev_idx = prev_tokens.index(token)
            prev_pos = prev_idx + 1
            curr_pos = i + 1
            prev_score = prev_scores[prev_idx]
            curr_score = curr_scores[i]
            
            if prev_pos != curr_pos or prev_score != curr_score:
                changes["position_changes"].append({
                    "token": token,
                    "prev_position": prev_pos,
                    "curr_position": curr_pos,
                    "prev_score": prev_score,
                    "curr_score": curr_score,
                    "position_delta": curr_pos - prev_pos
                })
            else:
                changes["unchanged"].append({
                    "token": token,
                    "position": curr_pos,
                    "score": curr_score
                })
    
    return changes

def format_telegram_notification(changes: Dict, current: List[Tuple[str, int]], previous: Dict) -> str:
    """Format changes into a beautiful Telegram message"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    
    message = f"🔔 <b>TOP 6 CHANGE DETECTED!</b>\n"
    message += f"⏰ Time: {timestamp}\n\n"
    
    # New entries
    if changes["new_entries"]:
        message += "📈 <b>NEW ENTRIES:</b>\n"
        for entry in changes["new_entries"]:
            message += f"  • <b>{entry['token']}</b> entered at #{entry['position']} (Score: {entry['score']})\n"
        message += "\n"
    
    # Dropped out
    if changes["dropped_out"]:
        message += "📉 <b>DROPPED OUT:</b>\n"
        for entry in changes["dropped_out"]:
            message += f"  • <b>{entry['token']}</b> dropped from #{entry['position']} (Score: {entry['score']})\n"
        message += "\n"
    
    # Position changes
    if changes["position_changes"]:
        message += "🔄 <b>POSITION CHANGES:</b>\n"
        for change in changes["position_changes"]:
            arrow = "⬆️" if change['position_delta'] < 0 else "⬇️"
            delta_str = f"({change['position_delta']:+d})" if change['position_delta'] != 0 else ""
            score_change = f"{change['prev_score']} → {change['curr_score']}" if change['prev_score'] != change['curr_score'] else f"Score: {change['curr_score']}"
            
            if change['position_delta'] != 0:
                message += f"  • <b>{change['token']}</b>: #{change['prev_position']} → #{change['curr_position']} {delta_str} {arrow}\n"
                message += f"    {score_change}\n"
            else:
                # Position same but score changed
                message += f"  • <b>{change['token']}</b> at #{change['curr_position']}: {score_change}\n"
        message += "\n"
    
    # Unchanged (optional, can comment out if too verbose)
    if changes["unchanged"]:
        message += "✅ <b>UNCHANGED:</b>\n"
        for entry in changes["unchanged"]:
            message += f"  • <b>{entry['token']}</b> at #{entry['position']} (Score: {entry['score']})\n"
        message += "\n"
    
    # Summary
    message += "━━━━━━━━━━━━━━━━━━━━━━\n"
    prev_tokens = ', '.join(previous['tokens'])
    curr_tokens = ', '.join([token for token, score in current])
    message += f"<b>Previous:</b> {prev_tokens}\n"
    message += f"<b>Current:</b>  {curr_tokens}"
    
    return message

# ===================== RESULTS MANAGEMENT =====================

def save_results(top6: List[Tuple[str, int]]):
    """Save top 6 results to JSON file with historical tracking"""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    
    # Load existing history
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = {}
    
    # Add today's results
    history[today] = {
        "tokens": [{"token": token, "score": score} for token, score in top6],
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Keep only last MAX_HISTORY_DAYS
    sorted_dates = sorted(history.keys(), reverse=True)
    if len(sorted_dates) > MAX_HISTORY_DAYS:
        for old_date in sorted_dates[MAX_HISTORY_DAYS:]:
            del history[old_date]
    
    # Save to file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Results saved to {RESULTS_FILE}")

def display_results(top6: List[Tuple[str, int]], volume_data: Dict):
    """Display results in a clean, readable format"""
    today = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    print("\n" + "=" * 60)
    print(f"🏆 TOP 6 TOKENS - {today}")
    print("=" * 60)
    
    for rank, (token, score) in enumerate(top6, 1):
        volume = volume_data.get(token, 0)
        print(f"#{rank}  {token:8s}  |  Score: {score:3d}  |  24h Volume: ${volume:,.0f}")
    
    print("=" * 60)

def display_history():
    """Display historical results if available"""
    if not os.path.exists(RESULTS_FILE):
        return
    
    with open(RESULTS_FILE, 'r') as f:
        history = json.load(f)
    
    if not history:
        return
    
    print("\n" + "=" * 60)
    print("📊 HISTORICAL TOP 6")
    print("=" * 60)
    
    # Show last 7 days
    sorted_dates = sorted(history.keys(), reverse=True)[:7]
    
    for date in sorted_dates:
        tokens = [item['token'] for item in history[date]['tokens']]
        print(f"{date}: {', '.join(tokens)}")
    
    print("=" * 60)

# ===================== MAIN EXECUTION =====================

def main():
    print("=" * 60)
    print("🚀 HYPERLIQUID TOKEN RANKING SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Volume threshold: ${VOLUME_THRESHOLD:,}")
    print(f"RSI period: {RSI_PERIOD}, MA period: {MA_PERIOD}")
    
    # Load previous Top 6 for comparison
    previous_top6 = load_last_top6()
    
    # Step 1: Fetch all perpetuals
    print("\n📡 Step 1: Fetching perpetual tokens...")
    all_tokens = fetch_perpetuals_meta()
    
    if not all_tokens:
        print("❌ Failed to fetch tokens. Exiting.")
        return
    
    # Step 2: Fetch volume data and filter
    print("\n📡 Step 2: Fetching 24h volume data...")
    volume_data = fetch_asset_contexts()
    
    if not volume_data:
        print("❌ Failed to fetch volume data. Exiting.")
        return
    
    # Filter tokens by volume
    filtered_tokens = [
        token for token in all_tokens 
        if volume_data.get(token, 0) > VOLUME_THRESHOLD
    ]
    
    print(f"✅ {len(filtered_tokens)} tokens passed volume filter (> ${VOLUME_THRESHOLD:,})")
    print(f"   Tokens: {', '.join(filtered_tokens[:10])}{'...' if len(filtered_tokens) > 10 else ''}")
    
    if len(filtered_tokens) < 2:
        print("❌ Not enough tokens to analyze. Exiting.")
        return
    
    # Step 3: Fetch historical data for all filtered tokens
    print(f"\n📡 Step 3: Fetching {LOOKBACK_DAYS} days of price data...")
    data_cache = {}
    
    for i, token in enumerate(filtered_tokens):
        df = fetch_daily_candles(token, LOOKBACK_DAYS)
        if not df.empty:
            data_cache[token] = df
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Fetched data for {i + 1}/{len(filtered_tokens)} tokens...")
        
        # Rate limiting
        time.sleep(0.1)
    
    print(f"✅ Successfully fetched data for {len(data_cache)} tokens")
    
    # Step 4: Calculate scores
    print("\n🔬 Step 4: Running ratio analysis and scoring...")
    scores = calculate_scores(list(data_cache.keys()), data_cache)
    
    # Step 5: Get top 6
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top6 = sorted_scores[:6]
    
    # Step 6: Display results
    display_results(top6, volume_data)
    
    # Step 7: Detect changes and send notification
    print("\n🔍 Step 7: Detecting changes...")
    changes = detect_changes(previous_top6, top6)
    
    if changes:
        print("🔔 Changes detected! Sending Telegram notification...")
        message = format_telegram_notification(changes, top6, previous_top6)
        send_telegram_message(message)
    else:
        print("✅ No changes detected. No notification sent.")
    
    # Step 8: Save results
    save_results(top6)
    save_last_top6(top6)
    display_history()
    
    print(f"\n✅ Analysis completed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

if __name__ == "__main__":
    main()
