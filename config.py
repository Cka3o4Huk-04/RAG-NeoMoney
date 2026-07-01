"""
Модуль конфигурации и валидации настроек приложения.

Централизованное управление конфигурацией и валидация переменных окружения.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Конфигурация приложения."""
    
    # API настройки
    use_proxyapi: bool = False
    openai_api_key: Optional[str] = None
    proxyapi_key: Optional[str] = None
    proxyapi_base_url: str = "https://api.proxyapi.ru/openai/v1"
    
    # Модели
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    
    # Хранилища
    chroma_persist_dir: str = "./chroma_db"
    cache_file: str = "cache.json"
    logs_db_file: str = "logs.db"
    
    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_admin_id: Optional[str] = None
    
    @property
    def api_key(self) -> Optional[str]:
        """Возвращает активный API ключ в зависимости от провайдера."""
        return self.proxyapi_key if self.use_proxyapi else self.openai_api_key
    
    @property
    def base_url(self) -> Optional[str]:
        """Возвращает базовый URL API (None для OpenAI)."""
        return self.proxyapi_base_url if self.use_proxyapi else None
    
    @property
    def provider_name(self) -> str:
        """Возвращает название текущего провайдера."""
        return "ProxyAPI" if self.use_proxyapi else "OpenAI"
    
    def validate(self) -> bool:
        """
        Проверяет корректность конфигурации.
        
        Returns:
            True если конфигурация валидна, False иначе
        """
        if self.use_proxyapi:
            if not self.proxyapi_key:
                logger.error("USE_PROXYAPI=true, но PROXYAPI_KEY не задан")
                return False
        else:
            if not self.openai_api_key:
                logger.error("USE_PROXYAPI=false, но OPENAI_API_KEY не задан")
                return False
        
        if self.temperature < 0 or self.temperature > 2:
            logger.error("TEMPERATURE должен быть между 0 и 2")
            return False
        
        return True
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> 'AppConfig':
        """
        Создает конфигурацию из переменных окружения.
        
        Args:
            env_file: Путь к файлу .env (если None, используется .env из текущей директории)
        
        Returns:
            AppConfig с загруженными настройками
        """
        # Загружаем переменные окружения (только один раз)
        load_dotenv(dotenv_path=env_file)
        
        config = cls()
        
        # API настройки
        config.use_proxyapi = os.getenv("USE_PROXYAPI", "false").lower() == "true"
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        config.proxyapi_key = os.getenv("PROXYAPI_KEY")
        config.proxyapi_base_url = os.getenv("PROXYAPI_BASE_URL", config.proxyapi_base_url)
        
        # Модели
        config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
        config.llm_model = os.getenv("MODEL_NAME", config.llm_model)
        
        try:
            config.temperature = float(os.getenv("TEMPERATURE", str(config.temperature)))
        except ValueError:
            logger.warning(f"Неверное значение TEMPERATURE, используется {config.temperature}")
        
        # Хранилища
        config.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", config.chroma_persist_dir)
        config.cache_file = os.getenv("CACHE_FILE", config.cache_file)
        config.logs_db_file = os.getenv("LOGS_DB_FILE", config.logs_db_file)
        
        # Telegram
        config.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        config.telegram_admin_id = os.getenv("TELEGRAM_ADMIN_ID")
        
        return config
    
    def log_config(self) -> None:
        """Логирует текущую конфигурацию (без секретных данных)."""
        logger.info(f"Конфигурация приложения:")
        logger.info(f"  Провайдер API: {self.provider_name}")
        logger.info(f"  Модель эмбеддингов: {self.embedding_model}")
        logger.info(f"  Модель LLM: {self.llm_model}")
        logger.info(f"  Температура: {self.temperature}")
        logger.info(f"  Директория ChromaDB: {self.chroma_persist_dir}")
        logger.info(f"  Файл кеша: {self.cache_file}")
        logger.info(f"  Файл базы логов: {self.logs_db_file}")
        logger.info(f"  Telegram бот: {'настроен' if self.telegram_bot_token else 'не настроен'}")
        logger.info(f"  Telegram админ: {'настроен' if self.telegram_admin_id else 'не настроен'}")


# Глобальный экземпляр конфигурации
config: Optional[AppConfig] = None


def get_config(force_reload: bool = False) -> AppConfig:
    """
    Получает глобальную конфигурацию приложения.
    
    Args:
        force_reload: Принудительно перезагрузить конфигурацию
        
    Returns:
        AppConfig с текущими настройками
    """
    global config
    
    if config is None or force_reload:
        config = AppConfig.from_env()
        
        if not config.validate():
            raise ValueError("Неверная конфигурация приложения")
        
        config.log_config()
    
    return config
