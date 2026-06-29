"""
Модуль для реализации RAG (Retrieval-Augmented Generation).

RAG объединяет поиск релевантной информации (Retrieval) с генерацией ответа (Generation)
для создания более точных и информативных ответов на вопросы пользователя.
"""

import logging
import os
from typing import List, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class RAGAssistant:
    """
    Класс RAG-ассистента, который использует векторный поиск и LLM для ответов.
    
    Процесс работы:
    1. Получает запрос пользователя
    2. Ищет релевантные документы в векторной базе
    3. Формирует контекст из найденных документов
    4. Отправляет запрос + контекст в LLM
    5. Возвращает сгенерированный ответ
    """
    
    def __init__(
        self,
        vector_store,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Инициализация RAG-ассистента.
        
        Args:
            vector_store: Экземпляр VectorStore для поиска документов
            api_key: API ключ OpenAI/ProxyAPI (если None, берется из переменной окружения)
            base_url: Базовый URL API (для ProxyAPI: https://api.proxyapi.ru/openai/v1)
            model: Название модели для генерации ответов
            temperature: Параметр "креативности" модели (0.0 - детерминированный, 1.0 - креативный)
            system_prompt: Системный промпт (если None, используется промпт по умолчанию)
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        
        # Устанавливаем системный промпт
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = "Ты - полезный AI-ассистент, который отвечает на вопросы на основе предоставленного контекста."
        
        # Инициализируем клиент OpenAI
        # Поддерживаем как прямой OpenAI API, так и ProxyAPI через base_url
        client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
        
        provider = "ProxyAPI" if base_url else "OpenAI"
        logger.info(f"✓ RAG-ассистент инициализирован (провайдер: {provider}, модель: {model})")
    
    def _format_context(self, search_results: List[Tuple[str, str, float]]) -> str:
        """
        Форматирует результаты поиска в контекст для LLM.
        
        Args:
            search_results: Список результатов поиска (текст, источник, расстояние)
            
        Returns:
            Отформатированный текст контекста
        """
        if not search_results:
            return "Релевантных документов не найдено."
        
        context_parts = []
        
        for i, (chunk_text, source, distance) in enumerate(search_results, 1):
            context_parts.append(
                f"[Документ {i} - {source}]\n{chunk_text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Создает промпт для LLM, включающий контекст и запрос пользователя.
        
        Args:
            query: Запрос пользователя
            context: Контекст из найденных документов
            
        Returns:
            Сформированный промпт
        """
        prompt = f"""{self.system_prompt}

Используй следующую информацию из базы знаний, чтобы ответить на вопрос пользователя.

ВАЖНО: 
- Отвечай на основе предоставленного контекста
- Если в контексте нет информации для ответа, честно скажи об этом
- Отвечай на русском языке
- Будь конкретным и информативным

=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{query}

=== ОТВЕТ ===
"""
        return prompt
    
    def generate_response(
        self, 
        query: str, 
        top_k: int = 3,
        verbose: bool = True
    ) -> Tuple[str, List[Tuple[str, str, float]]]:
        """
        Генерирует ответ на запрос пользователя используя RAG.
        
        Это основной метод, который:
        1. Ищет релевантные документы
        2. Формирует контекст
        3. Отправляет запрос в LLM
        4. Возвращает ответ
        
        Args:
            query: Запрос пользователя
            top_k: Количество документов для поиска
            verbose: Выводить ли детальную информацию о процессе
            
        Returns:
            Кортеж (ответ_llm, список_найденных_документов)
        """
        # Шаг 1: Поиск релевантных документов в векторной базе
        if verbose:
            logger.info(f"Поиск релевантных документов (top_k={top_k})...")
        
        search_results = self.vector_store.search(query, top_k=top_k)
        
        if verbose and search_results:
            logger.info(f"Найдено {len(search_results)} релевантных фрагментов:")
            for i, (chunk, source, distance) in enumerate(search_results, 1):
                logger.info(f"  {i}. [{source}] (similarity: {1 - distance:.3f})")
                logger.info(f"     {chunk[:100]}...")
        
        # Шаг 2: Форматируем контекст из найденных документов
        context = self._format_context(search_results)
        
        # Шаг 3: Создаем промпт с контекстом и запросом
        prompt = self._create_prompt(query, context)
        
        # Шаг 4: Отправляем запрос в LLM
        if verbose:
            logger.info(f"Генерация ответа с помощью {self.model}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты - полезный AI-ассистент, который отвечает на вопросы на основе предоставленного контекста."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # Извлекаем текст ответа
            answer = response.choices[0].message.content.strip()
            
            return answer, search_results
            
        except Exception as e:
            error_message = f"Ошибка при генерации ответа: {str(e)}"
            logger.error(error_message)
            return error_message, search_results
    
    def simple_response(self, query: str) -> str:
        """
        Упрощенная версия generate_response, возвращающая только текст ответа.
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Ответ LLM
        """
        answer, _ = self.generate_response(query, verbose=False)
        return answer

