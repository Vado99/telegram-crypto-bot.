import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, CallbackContext
import ccxt
import pandas as pd
import ta  # Використання ta для технічного аналізу
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
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['upper_band'] = bb.bollinger_hband()
        df['middle_band'] = bb.bollinger_mavg()
        df['lower_band'] = bb.bollinger_lband()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch'] = stoch.stoch()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
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
        return None, None, None, None

# Основна функція команди /start
async def start(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="🚀 Привіт! Я ваш трейдинг бот. Використовуйте /trade для отримання прогнозу.")
    logger.info(f"Користувач {update.effective_user.username} почав взаємодію з ботом.")

# Основна функція команди /trade
async def trade(update: Update, context: CallbackContext):
    try:
        markets = get_all_markets()
        if not markets:
            await notify_error(context, "Не вдалося отримати ринки.")
            return

        symbol = markets[0]  # Вибір першої пари для демонстрації
        timeframe = '1h'
        df = get_market_data(symbol, timeframe)
        if df is None:
            await notify_error(context, "Не вдалося отримати ринкові дані.")
            return

        model, scaler = train_model(df)
        if model is None or scaler is None:
            await notify_error(context, "Не вдалося навчити модель.")
            return

        predicted_change = predict_price_change(model, scaler, df)
        current_price = df['close'].iloc[-1]
        stop_loss, tp1, tp2, tp3 = calculate_risk_management(current_price, predicted_change)

        message = (f"💹 Прогноз для {symbol}:\n"
                   f"🔮 Прогнозована зміна ціни: {predicted_change:.2f}%\n"
                   f"💰 Поточна ціна: {current_price:.2f}\n"
                   f"🛑 Stop-Loss: {stop_loss:.2f}\n"
                   f"🚀 Take-Profit 1: {tp1:.2f}\n"
                   f"🚀 Take-Profit 2: {tp2:.2f}\n"
                   f"🚀 Take-Profit 3: {tp3:.2f}\n")

        await context.bot.send_message(chat_id=update.effective_chat.id, text=message)
        logger.info(f"Успішно надіслано прогноз для {symbol}.")
    except Exception as e:
        await notify_error(context, f"Сталася помилка: {e}")

# Основна функція для налаштування бота
def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("trade", trade))

    application.run_polling()

if __name__ == '__main__':
    main()
