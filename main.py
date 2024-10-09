import logging
import ccxt
import pandas as pd
import pandas_ta as ta  # Використовуємо pandas_ta для індикаторів
import requests
from textblob import TextBlob
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext, JobQueue
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import TELEGRAM_BOT_TOKEN, API_KEY, API_SECRET, TWITTER_API_KEY

# Підключення до ByBit API
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})

# Логування
logging.basicConfig(level=logging.INFO)

# Команда для старту
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Привіт! Я бот для аналізу ринку криптовалют. Я готовий до роботи!")

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

# Додавання технічних індикаторів за допомогою pandas_ta
def add_indicators(df):
    # RSI
    df['rsi'] = df.ta.rsi(close=df['close'], length=14)

    # Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = df.ta.bbands(close=df['close'], length=20, std=2).values.T

    # MACD
    macd = df.ta.macd(close=df['close'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']

    # SMA (Simple Moving Average)
    df['sma_50'] = df.ta.sma(close=df['close'], length=50)
    df['sma_200'] = df.ta.sma(close=df['close'], length=200)

    # EMA (Exponential Moving Average)
    df['ema_50'] = df.ta.ema(close=df['close'], length=50)
    df['ema_200'] = df.ta.ema(close=df['close'], length=200)

    # ADX (Average Directional Index)
    adx = df.ta.adx()
    df['adx'] = adx['ADX_14']

    # Stochastic Oscillator
    stoch = df.ta.stoch()
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']

    return df

# Аналіз новин і настрою через Twitter
def analyze_sentiment(symbol):
    query = f"{symbol} crypto"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100"
    headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
    response = requests.get(url, headers=headers)
    tweets = response.json().get('data', [])
    sentiment_score = sum(TextBlob(tweet['text']).sentiment.polarity for tweet in tweets) / len(tweets) if tweets else 0
    return sentiment_score

# Навчання моделі ШІ та прогноз
def train_and_predict(df):
    df = add_indicators(df)
    df['price_change'] = df['close'].pct_change().shift(-1) * 100  # Прогноз у %
    X = df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'sma_50', 'ema_50', 'adx', 'stoch_k']].dropna()
    y = df['price_change'].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    X_new = scaler.transform(df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'sma_50', 'ema_50', 'adx', 'stoch_k']].tail(1))
    predicted_change = model.predict(X_new)[0]
    return predicted_change

# Розрахунок ризик-менеджменту та точок входу
def calculate_risk_management(price, predicted_change):
    stop_loss = price * (1 - 0.02)  # Стоп-лос на 2%
    take_profit_1 = price * (1 + predicted_change / 100 * 0.5)  # 50% від прогнозу
    take_profit_2 = price * (1 + predicted_change / 100 * 0.75)  # 75% від прогнозу
    take_profit_3 = price * (1 + predicted_change / 100)  # 100% від прогнозу
    return stop_loss, take_profit_1, take_profit_2, take_profit_3

# Функція для аналізу та відправки сигналів
async def analyze_and_send_signal(context: CallbackContext):
    symbols = ['BTC/USDT', 'ETH/USDT']  # Можна додати всі торгові пари з ByBit
    for symbol in symbols:
        df = get_market_data(symbol)
        if df is None:
            continue

        predicted_change = train_and_predict(df)
        if predicted_change >= 30:
            price = df['close'].iloc[-1]
            stop_loss, tp1, tp2, tp3 = calculate_risk_management(price, predicted_change)
            
            message = (
                f"Сигнал для {symbol}:\n"
                f"Ціна входу: {price:.2f}\n"
                f"Стоп-лос: {stop_loss:.2f}\n"
                f"Тейк-профіти: {tp1:.2f}, {tp2:.2f}, {tp3:.2f}\n"
                f"Прогнозована зміна: {predicted_change:.2f}%"
            )
            await context.bot.send_message(chat_id=context.job.context, text=message)

# Основна функція запуску бота
async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    
    # Планування завдань (кожні 5 хвилин)
    job_queue = application.job_queue
    job_queue.run_repeating(analyze_and_send_signal, interval=300)

    await application.start()
    await application.updater.start_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
