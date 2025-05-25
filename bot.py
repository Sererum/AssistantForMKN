from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from prototype import Assistant
import asyncio

TOKEN = "8176977256:AAFpkbhz2JctYDl71djHZORexTUnj_Q4Yqc"  # Замените на токен от @BotFather
BOT_USERNAME = "@BotAssistant122121bot"  # Например, @SimpleHelperBot

assistant = Assistant()

# Ответ на /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для ответа на ваши вопросы, касающиеся административной части. Напиши /help, чтобы узнать, что я умею.")

# Ответ на /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("У меня всего 2 команды:\n/start - начать общение\n/help - помощь\n\nЕсли хочешь узнать ответ на свой вопрос - просто напиши его в чат. На некоторые вопросы я не смогу ответить, если они касаются чего либо, кроме административной части, например я не смогу решить задачу по физике, но смогу ответить где находится дирекция ИКНК.\n\nНадеюсь ты найдешь ответы на все свои вопросы 🥰.")

# Обработчик текстовых сообщений
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    # Отправляем сообщение о поиске ответа
    searching_msg = await update.message.reply_text("Ищем ответ на ваш вопрос...")
    
    try:
        # Асинхронно обрабатываем запрос
        answer = await asyncio.to_thread(assistant.get_answer, question)
        # Редактируем сообщение с результатом
        await searching_msg.edit_text(answer)
    except Exception as e:
        # В случае ошибки обновляем сообщение об ошибке
        await searching_msg.edit_text(f"⚠️ Произошла ошибка: {str(e)}")

# Ответ на необрабатываемые сообщения
async def unknown_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Не понимаю, что это значит 😢")

# Запуск бота
if __name__ == "__main__":
    app = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков команд
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Регистрация обработчиков сообщений
    app.add_handler(MessageHandler(filters.TEXT, handle_question))
    app.add_handler(MessageHandler(filters.ALL, unknown_input))

    print("Бот запущен...")
    app.run_polling()
