"""
Модуль для обработки сложных вопросов, на которые RAG-система не может ответить.

Функционал:
1. Предложение пользователю переформулировать вопрос
2. Подсчет попыток уточнения
3. Отправка сложного вопроса администратору в Telegram (если настроено)
"""

import logging
import os
from typing import List, Tuple, Optional
from telegram_bot import TelegramRAGBot

logger = logging.getLogger(__name__)


class ClarificationHandler:
    """
    Класс для обработки сложных вопросов.
    
    Если RAG-система не может найти релевантный ответ:
    1. Предлагает пользователю переформулировать вопрос (до MAX_CLARIFICATION_ATTEMPTS раз)
    2. Если после всех попыток ответ не найден, отправляет вопрос администратору в Telegram
    """
    
    def __init__(self, max_attempts: int = 2, min_relevance_score: float = None):
        """
        Инициализация обработчика сложных вопросов.
        
        Args:
            max_attempts: Максимальное количество попыток уточнения вопроса
            min_relevance_score: Минимальный порог релевантности (0.0-1.0). 
                                 Если None, берется из переменной окружения MIN_RELEVANCE_SCORE
        """
        self.max_attempts = max_attempts
        if min_relevance_score is None:
            min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
        self.min_relevance_score = min_relevance_score
        
        # Хранилище попыток для каждого пользователя/сессии
        self.attempt_history = {}  # {user_id: [(question, best_score), ...]}
        
        logger.info(f"ClarificationHandler инициализирован (max_attempts={max_attempts}, min_score={min_relevance_score})")
    
    def is_question_clear(
        self, 
        search_results: List[Tuple[str, str, float]], 
        user_id: str = "default"
    ) -> bool:
        """
        Проверяет, достаточно ли релевантны найденные документы.
        
        Args:
            search_results: Результаты поиска (текст, источник, расстояние)
            user_id: ID пользователя/сессии
            
        Returns:
            True если вопрос ясен и есть релевантные документы, False иначе
        """
        if not search_results:
            return False
        
        # Получаем лучший результат (минимальное расстояние = максимальная релевантность)
        best_distance = min([distance for _, _, distance in search_results])
        best_score = 1.0 - best_distance  # Преобразуем расстояние в релевантность
        
        logger.info(f"Лучшая релевантность для пользователя {user_id}: {best_score:.3f}")
        
        return best_score >= self.min_relevance_score
    
    def needs_clarification(
        self, 
        search_results: List[Tuple[str, str, float]], 
        user_id: str = "default"
    def add_attempt(self, user_id: str, question: str, best_score: float):
        """
        Добавляет попытку уточнения вопроса.
        
        Args:
            user_id: ID пользователя/сессии
            question: Вопрос пользователя
            best_score: Лучшая релевантность найденных документов
        """
        if user_id not in self.attempt_history:
            self.attempt_history[user_id] = []
        
        self.attempt_history[user_id].append({
            'question': question,
            'best_score': best_score,
            'timestamp': None
        })
        
        logger.info(f"Добавлена попытка для {user_id}: '{question}' (score={best_score:.3f})")
    
    def get_attempt_count(self, user_id: str) -> int:
        """
        Получает количество попыток уточнения для пользователя.
        
        Args:
            user_id: ID пользователя/сессии
            
        Returns:
            Количество попыток
        """
        return len(self.attempt_history.get(user_id, []))
    
    def should_send_to_admin(self, user_id: str) -> bool:
        """
        Проверяет, пора ли отправить вопрос администратору.
        
        Args:
            user_id: ID пользователя/сессии
            
        Returns:
            True если достигнуто максимальное количество попыток
        """
        attempt_count = self.get_attempt_count(user_id)
        return attempt_count >= self.max_attempts
    
    def get_clarification_prompt(self, user_id: str) -> str:
        """
        Формирует подсказку для пользователя с просьбой переформулировать вопрос.
        
        Args:
            user_id: ID пользователя/сессии
            
        Returns:
            Текст подсказки
        """
        attempt_count = self.get_attempt_count(user_id)
        remaining_attempts = self.max_attempts - attempt_count
        
        prompts = [
            "🤔 Похоже, я не совсем понял ваш вопрос. Попробуйте переформулировать его более конкретно.",
            "💭 Мне сложно найти ответ. Можете задать вопрос по-другому?",
            "🔍 Не нашел достаточно информации. Попробуйте уточнить вопрос.",
            "📝 Вопрос требует уточнения. Попробуйте сформулировать иначе.",
        ]
        
        if remaining_attempts <= 0:
            return "⏰ К сожалению, я исчерпал попытки уточнения. Ваш вопрос будет отправлен специалисту."
        
        # Выбираем подсказку на основе номера попытки
        prompt_idx = min(attempt_count, len(prompts) - 1)
        prompt = prompts[prompt_idx]
        
        if remaining_attempts > 0:
            prompt += f"\n\nОсталось попыток: {remaining_attempts}"
        
        return prompt
    
    def get_question_history(self, user_id: str) -> List[dict]:
        """
        Получает историю попыток для пользователя.
        
        Args:
            user_id: ID пользователя/сессии
            
        Returns:
            Список попыток
    def format_question_report(self, user_id: str, original_question: str) -> str:
        """
        Форматирует отчет о сложных вопросах для отправки администратору.
        
        Args:
            user_id: ID пользователя/сессии
            original_question: Исходный вопрос
            
        Returns:
            Отформатированный отчет
        """
        history = self.get_question_history(user_id)
        
        report = f"🔔 Сложный вопрос от пользователя\n\n"
        report += f"👤 ID пользователя: {user_id}\n"
        report += f"❓ Исходный вопрос: {original_question}\n\n"
        
        if history:
            report += "📝 История уточнений:\n"
            for i, attempt in enumerate(history, 1):
                report += f"  {i}. {attempt['question']} (релевантность: {attempt['best_score']:.3f})\n"
        else:
            report += "⚠️ Попыток уточнения не было\n"
        
        report += "\n💬 Пожалуйста, помогите пользователю с этим вопросом."
        
        return report
    
    async def send_to_admin(
        self, 
        user_id: str, 
        original_question: str, 
        bot: Optional[TelegramRAGBot] = None
    ) -> bool:
        """
        Отправляет сложный вопрос администратору в Telegram.
        
        Args:
            user_id: ID пользователя/сессии
            original_question: Исходный вопрос
            bot: Экземпляр TelegramRAGBot для отправки сообщения
            
        Returns:
            True если отправка успешна, False иначе
        """
        admin_id = os.getenv("TELEGRAM_ADMIN_ID")
        
        if not admin_id:
            logger.warning("TELEGRAM_ADMIN_ID не настроен. Вопрос не будет отправлен администратору.")
            return False
        
        if not bot:
            logger.warning("Telegram бот не предоставлен. Вопрос не будет отправлен.")
            return False
        
        try:
            # Форматируем отчет
            report = self.format_question_report(user_id, original_question)
            
            # Отправляем администратору
            await bot.send_message(
                chat_id=int(admin_id),
                text=report
            )
            
            logger.info(f"Сложный вопрос отправлен администратору {admin_id}")
            
            # Очищаем историю попыток
            self.clear_history(user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при отправке вопроса администратору: {e}")
            return False
        """
        return self.attempt_history.get(user_id, [])
    
    def clear_history(self, user_id: str):
        """
        Очищает историю попыток для пользователя.
        
        Args:
            user_id: ID пользователя/сессии
        """
        if user_id in self.attempt_history:
            del self.attempt_history[user_id]
            logger.info(f"История для {user_id} очищена")
    ) -> bool:
        """
        Определяет, нужно ли уточнение вопроса.
        
        Args:
            search_results: Результаты поиска
            user_id: ID пользователя/сессии
            
        Returns:
            True если вопрос требует уточнения, False иначе
        """
        return not self.is_question_clear(search_results, user_id)