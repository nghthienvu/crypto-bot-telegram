import os
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import schedule
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
import telebot

# === Load biến môi trường ===
load_dotenv()
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

# === Khởi tạo Bot & Binance ===
bot = telebot.TeleBot(API_TOKEN)
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
logging.basicConfig(level=logging.INFO)

coins = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT"}
RSI_LOW = 30
STRENGTH_THRESHOLD = 3

# === Các hàm tính chỉ báo kỹ thuật ===
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def compute_bollinger(prices, window=20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower

def detect_pattern(prices):
    last3 = prices[-3:]
    if len(last3) < 3:
        return None
    if last3[0] > last3[1] < last3[2]:
        return "Bullish Reversal"
    if last3[0] < last3[1] > last3[2]:
        return "Bearish Reversal"
    return None

def detect_volume_spike(volumes):
    rolling_mean = volumes.rolling(window=10).mean()
    return volumes.iloc[-1] > 1.5 * rolling_mean.iloc[-1]

def fetch_data(symbol, interval="1d", limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "trades", "buy_base", "buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return pd.to_numeric(df["close"]), pd.to_numeric(df["volume"])
    except Exception as e:
        logging.error(f"Lỗi lấy dữ liệu {symbol}: {e}")
        return None, None

def get_live_price(symbol):
    try:
        data = client.get_symbol_ticker(symbol=symbol)
        return float(data["price"])
    except Exception as e:
        logging.error(f"Lỗi lấy giá {symbol}: {e}")
        return None

def plot_chart(prices, rsi, macd, signal, symbol):
    os.makedirs("charts", exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(prices, label="Giá")
    axs[1].plot(rsi, label="RSI", color="purple")
    axs[1].axhline(y=30, color="red", linestyle="--")
    axs[2].plot(macd, label="MACD", color="blue")
    axs[2].plot(signal, label="Signal", color="orange")

    for ax in axs:
        ax.legend()

    axs[0].set_title(f"{symbol} Price")
    axs[1].set_title("RSI")
    axs[2].set_title("MACD")
    plt.tight_layout()

    path = f"charts/{symbol}_chart.png"
    plt.savefig(path)
    plt.close()
    return path

def send_alert(message, image_path):
    try:
        with open(image_path, "rb") as photo:
            bot.send_photo(CHAT_ID, photo=photo, caption=message, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Lỗi gửi ảnh Telegram: {e}")

def check_signals():
    for name, symbol in coins.items():
        try:
            prices, volumes = fetch_data(symbol)
            if prices is None or volumes is None or len(prices) < 30:
                raise Exception("Không đủ dữ liệu")

            rsi = compute_rsi(prices).dropna()
            macd, signal_line = compute_macd(prices)
            upper, lower = compute_bollinger(prices)

            rsi_val = rsi.iloc[-1]
            macd_cross = macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]
            bollinger_hit = prices.iloc[-1] < lower.iloc[-1]
            pattern = detect_pattern(prices)
            vol_spike = detect_volume_spike(volumes)

            strength = sum([
                rsi_val < RSI_LOW,
                macd_cross,
                bollinger_hit,
                pattern is not None,
                vol_spike
            ])

            live_price = get_live_price(symbol)
            price_str = f"${live_price:.2f}" if live_price else "N/A"

            status = "🔥 TÍN HIỆU MẠNH" if strength >= STRENGTH_THRESHOLD else (
                     "🚀 TIỀM NĂNG" if strength == STRENGTH_THRESHOLD - 1 else
                     "ℹ️ Chưa rõ ràng")

            msg = (
                f"<b>{symbol}</b>\n"
                f"Giá: <b>{price_str}</b> | RSI = <b>{rsi_val:.2f}</b>\n"
                f"MACD Cross: {'✅' if macd_cross else '❌'} | Strength: {strength}/5\n"
                f"Status: <b>{status}</b>\n"
                f"Nến: {pattern or 'Không có'} | Volume spike: {'✅' if vol_spike else '❌'}"
            )

            chart = plot_chart(prices, rsi, macd, signal_line, symbol)
            send_alert(msg, chart)
            with open("signal_log.csv", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()},{symbol},{rsi_val:.2f},{macd_cross},{bollinger_hit},{pattern},{vol_spike},{strength}\n")
            if os.path.exists(chart):
                os.remove(chart)

        except Exception as e:
            logging.exception(e)
            bot.send_message(CHAT_ID, f"⚠️ Lỗi xử lý {symbol}: {e}")

@bot.message_handler(func=lambda msg: msg.chat.id == CHAT_ID)
def respond(msg):
    text = msg.text.lower()
    if "btc" in text:
        symbol = "BTCUSDT"
    elif "eth" in text:
        symbol = "ETHUSDT"
    elif "sol" in text:
        symbol = "SOLUSDT"
    else:
        bot.send_message(msg.chat.id, "🧠 Gõ BTC, ETH hoặc SOL để nhận phân tích.")
        return

    price = get_live_price(symbol)
    reply = f"{symbol}: ${price:.2f}" if price else "⚠️ Không lấy được giá Binance."
    bot.send_message(msg.chat.id, reply)

# === Lên lịch chạy ===
schedule.every(15).minutes.do(check_signals)

if __name__ == "__main__":
    logging.info("⏳ Bot đang chạy...")
    while True:
        schedule.run_pending()
        time.sleep(60)
