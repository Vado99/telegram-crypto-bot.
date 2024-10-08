import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import ccxt
import pandas as pd
import ta
import requests
from textblob import TextBlob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os  # Для отримання змінних середовища

# Отримання змінних середовища для Heroku
TELEGRAM_BOT_TOKEN = os.getenv('7775389672:AAH471OO5u6hUm6N7P5YtMtz-qAe2oCmIA0')
API_KEY = os.getenv('OqbKOtRMY01Ho8KBRw')
API_SECRET = os.getenv('BdOnFmYMNTWMrCOWEdYIOUh05gyLD5K5M7qd')
TWITTER_API_KEY = os.getenv('AAAAAAAAAAAAAAAAAAAAAM4QwQEAAAAAU4QHR9h5QhF%2BHPy%2BfLl7f18SvB4%3DxJujcd3GVOJwANTSnKBhn6cGih72Lki5UGd0a0881ZEjkPKaXc')

# Налаштування логування
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Підключення до біржі ByBit через CCXT
exchange = ccxt.bybit({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})

# Функція для надсилання повідомлення про помилку в Telegram
async def notify_error(context: CallbackContext, error_message: str):
    try:
        chat_id = context.job.context['chat_id'] if context and context.job and context.job.context else None
        if chat_id:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Помилка: {error_message}")
        logger.error(f"Надіслано повідомлення про помилку: {error_message}")
    except Exception as e:
        logger.error(f"Не вдалося надіслати повідомлення про помилку: {e}")

# Функція для отримання всіх ринків (торгових пар)
def get_all_markets():
    try:
        markets = exchange.load_markets()
        logger.info("Успішно отримано ринки.")
        return [symbol for symbol in markets if '/USDT' in symbol]  # Аналіз тільки пар до USDT
    except Exception as e:
        logger.error(f"Помилка отримання ринків: {e}")
        return []

# Функція для отримання ринкових даних
def get_market_data(symbol, timeframe):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logger.info(f"Успішно отримано ринкові дані для {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Помилка отримання ринкових даних для {symbol}: {e}")
        return None

# Додавання технічних індикаторів
def add_indicators(df):
    try:
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['stoch'] = talib.STOCH(df['high'], df['low'], df['close'])[0]
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        logger.info("Технічні індикатори успішно додані.")
        return df
    except Exception as e:
        logger.error(f"Помилка при додаванні індикаторів: {e}")
        return df

# Функція для навчання моделі ШІ на історичних даних
def train_model(df):
    try:
        df = add_indicators(df)
        df['price_change'] = df['close'].pct_change().shift(-1) * 100  # Прогноз зміни ціни у відсотках
        df = df.dropna()

        X = df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'stoch', 'cci', 'atr']]
        y = df['price_change']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

        # Використання моделі RandomForestRegressor для прогнозування
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        r_squared = model.score(X_test, y_test)
        logger.info(f"R² моделі: {r_squared * 100:.2f}%")

        return model, scaler
    except Exception as e:
        logger.error(f"Помилка навчання моделі: {e}")
        return None, None

# Функція для прогнозування відсоткової зміни ціни
def predict_price_change(model, scaler, df):
    try:
        df = add_indicators(df)
        X_new = df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'stoch', 'cci', 'atr']].tail(1).dropna()
        X_scaled = scaler.transform(X_new)

        prediction = model.predict(X_scaled)[0]
        return prediction
    except Exception as e:
        logger.error(f"Помилка прогнозування зміни ціни: {e}")
        return 0

# Функція для аналізу настрою ринку
def analyze_sentiment(symbol):
    try:
        query = f"{symbol} crypto"
        url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100"
        headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
        response = requests.get(url, headers=headers)
        tweets = response.json().get('data', [])

        sentiment_score = 0
        for tweet in tweets:
            analysis = TextBlob(tweet['text'])
            sentiment_score += analysis.sentiment.polarity

        return sentiment_score / len(tweets) if tweets else 0
    except Exception as e:
        logger.error(f"Помилка аналізу настрою ринку для {symbol}: {e}")
        return 0

# Функція для отримання твітів про великі транзакції через Twitter API (@whale_alert)
def get_whale_transactions():
    try:
        url = f"https://api.twitter.com/2/tweets/search/recent?query=from:whale_alert"
        headers = {"Authorization": f"Bearer {TWITTER_API_KEY}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json().get('data', [])
            whale_transactions = []
            for tweet in data:
                if "transferred" in tweet['text']:
                    whale_transactions.append(tweet['text'])
            logger.info("Успішно отримано твітів про великі транзакції.")
            return whale_transactions
        else:
            logger.error(f"Помилка отримання твітів {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Помилка отримання твітів від Whale Alert: {e}")
        return []

# Функція для розрахунку ризик-менеджменту
def calculate_risk_management(price, predicted_change):
    try:
        stop_loss = price * (1 - 0.02)  # Stop-Loss на 2% нижче поточної ціни
        # Розрахунок кількох рівнів Take-Profit
        take_profit_1 = price * (1 + predicted_change / 100 * 0.5)  # Перший рівень Take-Profit (50% від прогнозу)
        take_profit_2 = price * (1 + predicted_change / 100 * 0.75)  # Другий рівень Take-Profit (75% від прогнозу)
        take_profit_3 = price * (1 + predicted_change / 100)  # Третій рівень Take-Profit (100% від прогнозу)
        return stop_loss, take_profit_1, take_profit_2, take_profit_3
    except Exception as e:
        logger.error(f"Помилка при розрахунку ризик-менеджменту: {e}")
        return 0, 0, 0, 0

# Функція для аналізу ринку та надсилання сигналів
async def analyze_market_and_send_signals(context: CallbackContext):
    try:
        symbols = get_all_markets()
        for symbol in symbols:
            df = get_market_data(symbol, '5m')  # Аналіз кожні 5 хвилин
            if df is not None and len(df) > 30:
                model, scaler = train_model(df)
                if model and scaler:
                    predicted_change = predict_price_change(model, scaler, df)
                    current_price = df['close'].iloc[-1]

                    # Перевірка, чи очікуваний профіт більше ніж 30%
                    if predicted_change > 30:
                        sentiment_score = analyze_sentiment(symbol)
                        whale_transactions = get_whale_transactions()

                        message = (f"💹 Сигнал для {symbol}:\n"
                                   f"📈 Прогнозована зміна ціни: {predicted_change:.2f}%\n"
                                   f"📊 Настрій ринку: {'Позитивний' if sentiment_score > 0 else 'Негативний'}\n"
                                   f"🐋 Whale Transactions: {'Так' if whale_transactions else 'Ні'}\n"
                                   f"🚫 Stop-Loss: {stop_loss:.2f}\n"
                                   f"💰 Take-Profit 1: {take_profit_1:.2f}\n"
                                   f"💰 Take-Profit 2: {take_profit_2:.2f}\n"
                                   f"💰 Take-Profit 3: {take_profit_3:.2f}\n")
                        await context.bot.send_message(chat_id=context.job.context['chat_id'], text=message)
                        logger.info(f"Сигнал надіслано для {symbol}.")
    except Exception as e:
        logger.error(f"Помилка в аналізі ринку та надсиланні сигналів: {e}")
        await notify_error(context, str(e))

# Команда для запуску аналізу ринку
async def start_analysis(update: Update, context: CallbackContext):
    try:
        chat_id = update.effective_chat.id
        context.job_queue.run_repeating(analyze_market_and_send_signals, interval=180, context={'chat_id': chat_id})  # Кожні 3 хвилини
        await update.message.reply_text("🔍 Аналіз ринку розпочато. Сигнали будуть надсилатися кожні 3 хвилини, якщо виявиться суттєва зміна ринку.")
    except Exception as e:
        logger.error(f"Помилка при запуску аналізу: {e}")
        await notify_error(context, str(e))

# Головна функція для запуску бота
async def main():
    try:
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        start_handler = CommandHandler('start', start_analysis)
        application.add_handler(start_handler)

        await application.start()
        await application.idle()
    except Exception as e:
        logger.error(f"Глобальна помилка: {e}")
        await notify_error(None, str(e))

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
