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
import os  # Для отримання змінних середовища

# Отримання змінних середовища для Heroku
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')

# Налаштування логування
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Підключення до біржі ByBit через CCXT
logger.info("Підключення до біржі ByBit через CCXT.")
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
        logger.info("Отримання всіх ринків.")
        markets = exchange.load_markets()
        logger.info(f"Успішно отримано ринки: {markets}")
        return [symbol for symbol in markets if '/USDT' in symbol]  # Аналіз тільки пар до USDT
    except Exception as e:
        logger.error(f"Помилка отримання ринків: {e}")
        return []

# Функція для отримання ринкових даних
def get_market_data(symbol, timeframe):
    try:
        logger.info(f"Отримання ринкових даних для {symbol} за таймфреймом {timeframe}.")
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
        logger.info("Додавання технічних індикаторів.")
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
        logger.info("Навчання моделі на історичних даних.")
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
        logger.info("Прогнозування відсоткової зміни ціни.")
        df = add_indicators(df)
        X_new = df[['rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal', 'stoch', 'cci', 'atr']].tail(1).dropna()
        X_scaled = scaler.transform(X_new)

        prediction = model.predict(X_scaled)[0]
        logger.info(f"Прогнозована зміна ціни: {prediction:.2f}%")
        return prediction
    except Exception as e:
        logger.error(f"Помилка прогнозування зміни ціни: {e}")
        return 0

# Функція для розрахунку ризик-менеджменту
def calculate_risk_management(price, predicted_change):
    try:
        logger.info("Розрахунок ризик-менеджменту.")
        stop_loss = price * (1 - 0.02)  # Stop-Loss на 2% нижче поточної ціни
        # Розрахунок кількох рівнів Take-Profit
        take_profit_1 = price * (1 + predicted_change / 100 * 0.5)  # Перший рівень Take-Profit (50% від прогнозу)
        take_profit_2 = price * (1 + predicted_change / 100 * 0.75)  # Другий рівень Take-Profit (75% від прогнозу)
        take_profit_3 = price * (1 + predicted_change / 100)  # Третій рівень Take-Profit (100% від прогнозу)
        logger.info(f"Ризик-менеджмент: Stop-Loss: {stop_loss:.2f}, Take-Profit 1: {take_profit_1:.2f}, Take-Profit 2: {take_profit_2:.2f}, Take-Profit 3: {take_profit_3:.2f}")
        return stop_loss, take_profit_1, take_profit_2, take_profit_3
    except Exception as e:
        logger.error(f"Помилка при розрахунку ризик-менеджменту: {e}")
        return None, None, None, None

# Основна функція команди /start
async def start(update: Update, context: CallbackContext):
    logger.info(f"Користувач {update.effective_user.username} почав взаємодію з ботом.")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="🚀 Привіт! Я ваш трейдинг бот. Використовуйте /trade для отримання прогнозу.")

# Основна функція команди /trade
async def trade(update: Update, context: CallbackContext):
    try:
        logger.info("Команда /trade отримана.")
        markets = get_all_markets()
        if not markets:
            logger.error("Не вдалося отримати ринки.")
            await notify_error(context, "Не вдалося отримати ринки.")
            return

        symbol = markets[0]  # Вибір першої пари для демонстрації
        timeframe = '1h'
        df = get_market_data(symbol, timeframe)
        if df is None:
            logger.error("Не вдалося отримати ринкові дані.")
            await notify_error(context, "Не вдалося отримати ринкові дані.")
            return

        # Додайте тут ваш код для прогнозування та ризик-менеджменту
        model, scaler = train_model(df)
        if model is None:
            await notify_error(context, "Не вдалося навчити модель.")
            return

        predicted_change = predict_price_change(model, scaler, df)
        stop_loss, take_profit_1, take_profit_2, take_profit_3 = calculate_risk_management(df['close'].iloc[-1], predicted_change)

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🔍 Прогноз зміни ціни для {symbol}: {predicted_change:.2f}%\n"
                 f"📉 Stop-Loss: {stop_loss:.2f}\n"
                 f"💰 Take-Profit 1: {take_profit_1:.2f}\n"
                 f"💰 Take-Profit 2: {take_profit_2:.2f}\n"
                 f"💰 Take-Profit 3: {take_profit_3:.2f}"
        )

    except Exception as e:
        logger.error(f"Помилка в команді /trade: {e}")
        await notify_error(context, f"Помилка в команді /trade: {e}")

# Додайте код для запуску бота, якщо це необхідно
# application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
# application.add_handler(CommandHandler("start", start))
# application.add_handler(CommandHandler("trade", trade))
# application.run_polling()
