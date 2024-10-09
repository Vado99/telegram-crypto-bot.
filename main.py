import logging
import ccxt
import pandas as pd
import pandas_ta as ta  # Використовуємо pandas_ta для індикаторів
import requests
from textblob import TextBlob
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor  # Додано для нейронної мережі

# Ваш API-токен для Telegram
TELEGRAM_BOT_TOKEN = 'your_telegram_bot_token'
API_KEY = 'your_bybit_api_key'
API_SECRET = 'your_bybit_api_secret'
TWITTER_API_KEY = 'your_twitter_api_key'

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
    return ticker['quoteVolume']  # Отримуємо обсяг у USDT

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
    
    # SMA
    df['sma'] = df.ta.sma(close=df['close'], length=50)
    
    # EMA
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

# Функція для пошуку схожих ситуацій
def find_similar_situations(df, current_price):
    df['price_change'] = df['close'].pct_change()
    current_change = df['price_change'].iloc[-1] if not df['price_change'].isnull().all() else 0

    # Порівнюємо останні 100 значень з поточним зміною
    similar_situations = df.iloc[-101:-1][(df['price_change'] - current_change).abs() < 0.01]
    return similar_situations

# Прогнозування зміни ціни за допомогою ШІ
def train_and_predict(df):
    df = add_indicators(df)
    df['price_change'] = df['close'].pct_change().shift(-1) * 100  # Прогноз у %
    
    # Видалення рядків з NaN
    df.dropna(inplace=True)

    if df.empty:
        return 0  # Повертаємо 0, якщо немає даних для прогнозування

    X = df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'sma', 'ema']]
    y = df['price_change']

    # Нормалізація даних
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Розбивка на навчальні та тестові дані
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    
    # Використання нейронної мережі
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)
    
    # Прогнозування
    X_new = scaler.transform(df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'sma', 'ema']].tail(1))
    predicted_change = model.predict(X_new)[0]
    return predicted_change

# Розрахунок точок входу, стоп-лосу та профітів
def calculate_risk_management(price, predicted_change):
    stop_loss = price * (1 - 0.02)  # Стоп-лос на 2%
    take_profit_1 = price * (1 + predicted_change / 100 * 0.5)  # 50% від прогнозу
    take_profit_2 = price * (1 + predicted_change / 100 * 0.75)  # 75% від прогнозу
    take_profit_3 = price * (1 + predicted_change / 100)  # 100% від прогнозу
    return stop_loss, take_profit_1, take_profit_2, take_profit_3

# Функція для аналізу та відправки сигналів
async def analyze_and_send_signal(context: CallbackContext):
    symbols = ['BTC/USDT', 'ETH/USDT']  # Додаємо потрібні торгові пари
    for symbol in symbols:
        df = get_market_data(symbol)
        if df is None or df.empty:
            continue

        # Прогнозування зміни ціни
        predicted_change = train_and_predict(df)
        
        # Пошук схожих ситуацій
        similar_situations = find_similar_situations(df, df['close'].iloc[-1])
        
        # Отримання обсягу торгів
        trade_volume = get_trade_volume(symbol)

        # Генерація сигналів
        if predicted_change >= 30:
            price = df['close'].iloc[-1]
            stop_loss, tp1, tp2, tp3 = calculate_risk_management(price, predicted_change)
            
            # Обробка ситуації, якщо схожі ситуації недостатні
            similar_situations_msg = similar_situations[['timestamp', 'close', 'price_change']].tail(5).to_string() if not similar_situations.empty else "Немає схожих ситуацій."
            
            message = (
                f"Сигнал для {symbol}:\n"
                f"Ціна входу: {price:.2f}\n"
                f"Стоп-лос: {stop_loss:.2f}\n"
                f"Тейк-профіти: {tp1:.2f}, {tp2:.2f}, {tp3:.2f}\n"
                f"Прогнозована зміна: {predicted_change:.2f}%\n"
                f"Обсяг торгів: {trade_volume:.2f} USDT\n"
                f"Схожі ситуації:\n{similar_situations_msg}"
            )
            await context.bot.send_message(chat_id=context.job.context, text=message)

# Основна функція запуску
async def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    
    # Планування завдань (кожні 5 хвилин)
    job_queue = application.job_queue
    job_queue.run_repeating(analyze_and_send_signal, interval=300, first=0)

    await application.start()
    await application.updater.start_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
