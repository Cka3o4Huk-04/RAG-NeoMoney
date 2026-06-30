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
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
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
        logger: DatabaseLogger
    ):
        """
        Инициализация Telegram бота.
        
        Args:
            token: Токен Telegram бота от @BotFather
            rag_assistant: Экземпляр RAG-ассистента
            cache: Экземпляр кеша ответов
            logger: Экземпляр логгера базы данных
        """
        self.rag_assistant = rag_assistant
        self.cache = cache
        self.logger = logger
        
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
            log_stats = self.logger.get_stats()
            
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
    
    async def logs_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /logs - экспорт логов в CSV"""
        try:
            user_id = str(update.effective_user.id)
            
            # Экспортируем логи текущего пользователя
            csv_content = self.logger.export_to_csv(user_id=user_id)
            
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
                answer, _ = self.rag_assistant.generate_response(
                    query=user_message,
                    top_k=3,
                    verbose=False
                )
                
                # Сохраняем в кеш
                self.cache.set(user_message, answer)
            
            # Вычисляем время ответа
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Логируем взаимодействие
            self.logger.log_interaction(
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
        
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            error_message = f"❌ Произошла ошибка при обработке запроса: {str(e)}"
            await update.message.reply_text(error_message)
            
            # Логируем ошибку
            self.logger.log_interaction(
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

