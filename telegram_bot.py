"""
Модуль для Telegram бота, интегрир��ванного с RAG-ассистентом.

Бот позволяет пользователям задава��ь вопросы ассистенту через Telegram
и получать ответы на основе векторного поиска и LLM.
"""

import logging
import os
import signal
import sys
import time
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger

logger = logging.getLogger(__name__)


class TelegramRAGBot:
    """
    Telegram бот для RAG-ассистента.
    
    Обрабатывает команды и сообщения от пользователей,
    логирует все взаимодействия в базу данных.
    """
    
    def __init__(
        self,
        token: str,
        rag_assistant: RAGAssistant,
        cache: ResponseCache,
        logger: DatabaseLogger,
        admin_id: Optional[str] = None
    ):
        """
        Инициализация Telegram бота.
        
        Args:
            token: Токен Telegram бота от @BotFather
            rag_assistant: Экземпляр RAG-ассистента
            cache: Экземпляр кеша ответов
            logger: Экземпляр логгера базы данных
            admin_id: ID администратора для эскалации вопросов
        """
        self.rag_assistant = rag_assistant
        self.cache = cache
        self.logger_db = logger
        self.admin_id = admin_id
        
        # Словарь для хранения ожидающих подтверждения эскалации
        # {user_id: {"query": query, "message_id": message_id}}
        self.escalation_pending = {}
        
        # Создаем приложение Telegram
        self.application = Application.builder().token(token).build()
        
        # Регистрируем обработчики команд
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("logs", self.logs_command))
        
        # Регистрируем обработчик текстовых сообщений
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        
        # Регистрируем обработчик callback-запросов (для кнопок эскалации)
        self.application.add_handler(
            CallbackQueryHandler(self.handle_escalation_callback)
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        try:
            # Генерируем приветственное сообщение на основе системного промпта
            welcome_message = self.rag_assistant.generate_welcome_message()
        except Exception as e:
            logger.error(f"Ошибка при генерации приветственного сообщения: {e}")
            welcome_message = """Добро пожаловать! 🚀

Я — ваш ассистент по адаптации в NeoMoney. Задавайте вопросы о процессах компании, HR-политиках, IT-инструментах и других аспектах работы!

📋 Доступные команды:
/help - подробная справка
/stats - статистика системы
/logs - история ваших запросов

💬 Просто напишите ваш вопрос!"""
        
        await update.message.reply_text(welcome_message)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        try:
            # Генерируем справку на основе системного промпта
            help_text = self.rag_assistant.generate_help_message()
        except Exception as e:
            logger.error(f"Ошибка при генерации справки: {e}")
            help_text = """📚 Справка по использованию бота:

Я — ваш ассистент по адаптации в NeoMoney. Задавайте вопросы о процессах компании, HR-политиках, IT-инструментах и других аспектах работы!

💡 Как это работает:
• Просто напишите вопрос — я найду ответ в Базе Знаний
• Ответы формируются на основе официальных документов компании
• Кеширование обеспечивает быструю работу

📋 Доступные команды:
/start - приветственное сообщение
/help - эта справка
/stats - статистика системы
/logs - история ваших запросов (CSV)

💬 Примеры вопросов:
• "Как настроить VPN?"
• "Где найти информацию о льготах?"
• "Как оформить отпуск?"
• "Какие документы нужны для трудоустройства?"
• "Как получить доступ к корпоративным системам?"
• "Какие есть правила безопасности?"
• "Как работает система оценки сотрудников?"
"""
        
        await update.message.reply_text(help_text)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        try:
            # Получаем статистику системы
            doc_count = self.rag_assistant.vector_store.collection.count()
            cache_size = self.cache.size()
            model = self.rag_assistant.model
            
            # Получаем статистику из логов
            log_stats = self.logger_db.get_stats()
            
            stats_message = f"""
📊 СТАТИСТИКА СИСТЕМЫ:

📚 База знаний:
  • Документов в ChromaDB: {doc_count}
  • Модель LLM: {model}

💾 Кеш:
  • Записей в кеше: {cache_size}

📝 Логи:
  • Всего запросов: {log_stats['total_requests']}
  • Из кеша: {log_stats['cached_requests']}
  • Уникальных пользователей: {log_stats['unique_users']}
  • Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс
            """
            
            await update.message.reply_text(stats_message.strip())
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            await update.message.reply_text(f"❌ Ошибка при получении статистики: {str(e)}")
    
    async def handle_escalation_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-запросов для эскалации вопросов администратору"""
        query = update.callback_query
        user = update.effective_user
        user_id = str(user.id)
        
        await query.answer()  # Показываем, что бот обработал нажатие
        
        if user_id not in self.escalation_pending:
            await query.edit_message_text("❌ Сессия эскалации истекла. Пожалуйста, задайте вопрос заново.")
            return
        
        escalation_data = self.escalation_pending[user_id]
        action = query.data
        
        if action == "escalate_yes":
            # Отправляем вопрос администратору
            try:
                admin_message = f"""
📨 НОВЫЙ ВОПРОС ОТ ПОЛЬЗОВАТЕЛЯ

👤 Пользователь: {escalation_data['username']} (ID: {user_id})
📝 Вопрос: {escalation_data['query']}
⏰ Время: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(escalation_data['timestamp']))}

Пожалуйста, ответьте пользователю напрямую в Telegram.
"""
                
                await context.bot.send_message(
                    chat_id=self.admin_id,
                    text=admin_message.strip()
                )
                
                # Обновляем сообщение пользователя
                await query.edit_message_text(
                    "✅ Ваш вопрос успешно отправлен администратору! Ожидайте ответа."
                )
                
                # Логируем эскалацию
                self.logger_db.log_interaction(
                    query=escalation_data['query'],
                    response="ESCALATED_TO_ADMIN",
                    source="telegram_escalation",
                    user_id=user_id,
                    username=escalation_data['username'],
                    from_cache=False,
                    response_time_ms=0
                )
                
                logger.info(f"Вопрос от {escalation_data['username']} эскалирован администратору")
                
            except Exception as e:
                logger.error(f"Ошибка при отправке вопроса администратору: {e}")
                await query.edit_message_text(
                    "❌ Произошла ошибка при отправке вопроса администратору. Попробуйте ещё раз."
                )
        
        elif action == "escalate_no":
            # Пользователь отказался от эскалации
            await query.edit_message_text(
                "👍 Хорошо! Если у вас появятся ещё вопросы — обращайтесь."
            )
        
        # Удаляем запись из pending
        del self.escalation_pending[user_id]
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /logs - экспорт логов в CSV"""
        try:
            user_id = str(update.effective_user.id)
            
            # Экспортируем логи текущего пользователя
            csv_content = self.logger_db.export_to_csv(user_id=user_id)
            
            if not csv_content:
                await update.message.reply_text(
                    "📝 Логов для вашего пользователя не найдено."
                )
                return
            
            # Сохраняем во временный файл
            filename = f"logs_{user_id}_{int(time.time())}.csv"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            # Отправляем файл пользователю
            with open(filename, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption="📊 Ваши логи взаимодействий с ботом"
                )
            
            # Удаляем временный файл
            os.remove(filename)
            
        except Exception as e:
            logger.error(f"Ошибка при экспорте логов: {e}")
            await update.message.reply_text(f"❌ Ошибка при экспорте логов: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений от пользователей"""
        user_message = update.message.text
        user = update.effective_user
        user_id = str(user.id)
        username = user.username or user.first_name or "Unknown"
        
        # Показываем, что бот печатает
        await update.message.chat.send_action(action="typing")
        
        start_time = time.time()
        
        try:
            # Проверяем кеш
            cached_answer = self.cache.get(user_message)
            from_cache = cached_answer is not None
            
            if cached_answer:
                answer = cached_answer
            else:
                # Выполняем RAG запрос
                answer, search_results = self.rag_assistant.generate_response(
                    query=user_message,
                    top_k=3,
                    verbose=False
                )
                
                # Проверяем, найден ли ответ (если search_results пустой - ответ не найден)
                answer_not_found = not search_results or "релевантных документов не найдено" in answer.lower()
                
                # Сохраняем в кеш
                self.cache.set(user_message, answer)
            
            # Вычисляем время ответа
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Логируем взаимодействие
            self.logger_db.log_interaction(
                query=user_message,
                response=answer,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=from_cache,
                response_time_ms=response_time_ms
            )
            
            # Отправляем ответ пользователю
            # Разбиваем длинные ответы на части (Telegram имеет лимит 4096 символов)
            max_length = 4000
            if len(answer) <= max_length:
                await update.message.reply_text(answer)
            else:
                # Отправляем частями
                parts = [answer[i:i+max_length] for i in range(0, len(answer), max_length)]
                for i, part in enumerate(parts):
                    if i == 0:
                        await update.message.reply_text(part)
                    else:
                        await update.message.reply_text(part)
            
            # Добавляем индикатор, если ответ из кеша
            if from_cache:
                await update.message.reply_text("💾 (ответ из кеша)", quote=False)
            
            # Если ответ не найден и настроен admin_id, предлагаем эскалацию
            if answer_not_found and self.admin_id:
                # Сохраняем вопрос для возможной эскалации
                self.escalation_pending[user_id] = {
                    "query": user_message,
                    "username": username,
                    "user_first_name": user.first_name,
                    "timestamp": time.time()
                }
                
                # Отправляем сообщение с предложением эскалации
                escalation_text = """
⚠️ К сожалению, я не нашёл ответ на ваш вопрос in базе знаний.

Хотите, чтобы я передал ваш вопрос администратору?
"""
                keyboard = [
                    [
                        InlineKeyboardButton("✅ Да, отправить администратору", callback_data="escalate_yes"),
                        InlineKeyboardButton("❌ Нет, спасибо", callback_data="escalate_no")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                escalation_msg = await update.message.reply_text(
                    escalation_text.strip(), 
                    reply_markup=reply_markup
                )
                
                # Сохраняем message_id для последующего редактирования/удаления
                self.escalation_pending[user_id]["message_id"] = escalation_msg.message_id
        
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            error_message = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            await update.message.reply_text(error_message)
            
            # Логируем ошибку
            self.logger_db.log_interaction(
                query=user_message,
                response=error_message,
                source="telegram",
                user_id=user_id,
                username=username,
                from_cache=False,
                response_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _setup_signal_handlers(self):
        """Настраивает обработчики сигналов для graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Получен сигнал остановки, завершаем работу...")
            self.shutdown()
            sys.exit(0)
        
        # Регистрируем обработчики для SIGINT и SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def shutdown(self):
        """Корректно завершает работу бота."""
        logger.info("Остановка Telegram бота...")
        if self.application and self.application.running:
            self.application.stop()
            self.application.shutdown()
        # Сохраняем кеш при завершении
        if self.cache:
            self.cache.flush()
        logger.info("Бот успешно остановлен")
    
    def run(self):
        """Запускает бота с обработкой graceful shutdown."""
        self._setup_signal_handlers()
        
        try:
            logger.info("🤖 Запуск Telegram бота...")
            logger.info("Бот готов к работе! Нажмите Ctrl+C для остановки.")
            self.application.run_polling()
        except KeyboardInterrupt:
            logger.info("Получен сигнал KeyboardInterrupt")
            self.shutdown()
        except Exception as e:
            logger.error(f"Ошибка при запуске бота: {e}")
            self.shutdown()
            raise

