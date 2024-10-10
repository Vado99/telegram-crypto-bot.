import logging
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
from textblob import TextBlob
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import openai

# Фікс для імпорту NaN із numpy
try:
    from numpy import nan as npNaN
except ImportError:
    from numpy import NaN as npNaN
    
# Ваш API-токен для Telegram
TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token'
API_KEY = 'your_bybit_api_key'
API_SECRET = 'your_bybit_api_secret'
TWITTER_API_KEY = 'your_twitter_api_key'
openai.api_key = "your_openai_api_key"  # API для GPT

# Підключення до ByBit API
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})

# Логування
logging.basicConfig(level=logging.INFO)

# Команда для старту
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привіт! Я аналітичний бот для трейдингу криптовалют. Я надсилатиму вам сигнали.")

# Команда для допомоги
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text("Ви можете використовувати такі команди:\n/start - Почати роботу\n/status - Перевірити статус\n/help - Отримати допомогу")

# Команда для перевірки статусу
async def status(update: Update, context: CallbackContext):
    await update.message.reply_text("Я працюю і аналізую ринок кожні 5 хвилин. Сигнали надсилаються за необхідності.")

# Отримання ринкових даних із ByBit
def get_market_data(symbol, timeframe='1h'):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Отримання обсягу торгів
def get_trade_volume(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker['quoteVolume']

# Додавання технічних індикаторів
def add_indicators(df):
    df['rsi'] = df.ta.rsi(close=df['close'], length=14)
    df['upper_band'], df['middle_band'], df['lower_band'] = df.ta.bbands(close=df['close'], length=20, std=2).values.T
    macd = df.ta.macd(close=df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['sma'] = df.ta.sma(close=df['close'], length=50)
    df['ema'] = df.ta.ema(close=df['close'], length=20)
    return df

# Аналіз настрою ринку через Twitter
def analyze_sentiment(symbol):
    query = f"{symbol} crypto"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100"
    headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
    response = requests.get(url, headers=headers)
    tweets = response.json().get('data', [])
    sentiment_score = sum(TextBlob(tweet['text']).sentiment.polarity for tweet in tweets) / len(tweets) if tweets else 0
    return sentiment_score

# GPT-3 для аналізу ринкових даних
def analyze_with_gpt(symbol, market_data, sentiment):
    prompt = f"""
    Analyze the following cryptocurrency data for {symbol}:
    - Market Data: {market_data}
    - Sentiment Analysis: {sentiment}
    
    Based on the analysis, provide a trading signal, including whether to buy, sell, or hold, and reasoning behind it.
    """
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.5
        )
        signal = response.choices[0].text.strip()
        return signal
    except Exception as e:
        logging.error(f"GPT API Error: {e}")
        return "Не вдалося отримати сигнал від ШІ"

# Прогнозування ціни за допомогою нейронної мережі
def train_and_predict(df):
    df = add_indicators(df)
    df['price_change'] = df['close'].pct_change().shift(-1) * 100
    df.dropna(inplace=True)
    if df.empty:
        return 0

    X = df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'sma', 'ema']]
    y = df['price_change']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)
    X_new = scaler.transform(df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'sma', 'ema']].tail(1))
    predicted_change = model.predict(X_new)[0]
    return predicted_change

# Розрахунок точок входу та профітів
def calculate_risk_management(price, predicted_change):
    stop_loss = price * (1 - 0.02)
    take_profit_1 = price * (1 + predicted_change / 100 * 0.5)
    take_profit_2 = price * (1 + predicted_change / 100 * 0.75)
    take_profit_3 = price * (1 + predicted_change / 100)
    return stop_loss, take_profit_1, take_profit_2, take_profit_3

# Аналіз і відправка сигналу з GPT-аналізом
async def analyze_and_send_signal_with_gpt(context: CallbackContext):
    symbols = ['BTC/USDT', 'ETH/USDT']
    for symbol in symbols:
        df = get_market_data(symbol)
        if df is None or df.empty:
            continue

        predicted_change = train_and_predict(df)
        trade_volume = get_trade_volume(symbol)
        sentiment_score = analyze_sentiment(symbol)
        market_data = df.tail(5).to_dict()
        gpt_signal = analyze_with_gpt(symbol, market_data, sentiment_score)

        message = (
            f"Сигнал для {symbol}:\n"
            f"GPT-аналіз: {gpt_signal}\n"
            f"Прогнозована зміна: {predicted_change:.2f}%\n"
            f"Обсяг торгів: {trade_volume:.2f} USDT\n"
            f"Настрій ринку: {sentiment_score:.2f}"
        )
        await context.bot.send_message(chat_id=context.job.context, text=message)

# Основна функція запуску бота
async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))

    job_queue = application.job_queue
    job_queue.run_repeating(analyze_and_send_signal_with_gpt, interval=300, first=0)

    await application.start()
    await application.updater.start_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
