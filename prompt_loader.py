"""
Модуль для загрузки системных промптов из файлов.

Промпты используются при запуске для настройки поведения ИИ-ассистента.
Каждый файл в папке prompts/ представляет отдельный промпт.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Класс для загрузки и управления системными промптами.
    
    Поддерживает загрузку промптов из файлов в директории prompts/.
    Каждый файл .txt содержит один промпт. Имя файла (без расширения)
    становится ключом промпта.
    """
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Инициализация загрузчика промптов.
        
        Args:
            prompts_dir: Директория с файлами промптов
        """
        self.prompts_dir = Path(prompts_dir)
        self.prompts: Dict[str, str] = {}
        
        logger.info(f"Инициализация загрузчика промптов: {prompts_dir}")
        
        # Создаем директорию, если она не существует
        if not self.prompts_dir.exists():
            logger.warning(f"Директория промптов не найдена: {prompts_dir}")
            logger.info("Создаю директорию prompts/ с примером промпта")
            self.prompts_dir.mkdir(exist_ok=True)
            self._create_default_prompt()
        
        # Загружаем промпты
        self.load_prompts()
    
    def _create_default_prompt(self):
        """Создает промпт по умолчанию, если папка пуста."""
        default_prompt = """Ты - полезный AI-ассистент, который отвечает на вопросы на основе предоставленного контекста.

Твои основные принципы:
1. Отвечай точно и по делу
2. Используй только информацию из предоставленного контекста
3. Если в контексте нет информации для ответа, честно скажи об этом
4. Отвечай на русском языке
5. Будь вежлив и профессионален

Формат ответов:
- Начинай с прямого ответа на вопрос
- При необходимости приводи детали и пояснения
- Указывай источники информации, если это уместно"""
        
        default_file = self.prompts_dir / "system_prompt.txt"
        with open(default_file, "w", encoding="utf-8") as f:
            f.write(default_prompt)
        
        logger.info(f"Создан промпт по умолчанию: {default_file}")
    
    def load_prompts(self) -> Dict[str, str]:
        """
        Загружает все промпты из директории.
        
        Returns:
            Словарь {ключ_промпта: текст_промпта}
        """
        self.prompts.clear()
        
        if not self.prompts_dir.exists():
            logger.error(f"Директория промптов не существует: {self.prompts_dir}")
            return self.prompts
        
        # Ищем файлы .txt
        txt_files = list(self.prompts_dir.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"Не найдено файлов промптов в {self.prompts_dir}")
            return self.prompts
        
        for file_path in txt_files:
            try:
                # Ключ промпта - имя файла без расширения
                prompt_key = file_path.stem
                
                # Читаем содержимое файла
                with open(file_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                
                if prompt_text:
                    self.prompts[prompt_key] = prompt_text
                    logger.info(f"Загружен промпт '{prompt_key}' ({len(prompt_text)} символов)")
                else:
                    logger.warning(f"Пустой файл промпта: {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Ошибка при загрузке промпта {file_path.name}: {e}")
        
        logger.info(f"Всего загружено промптов: {len(self.prompts)}")
        return self.prompts
    
    def get_prompt(self, key: str = "system_prompt") -> Optional[str]:
        """
        Получает промпт по ключу.
        
        Args:
            key: Ключ промпта (имя файла без расширения)
            
        Returns:
            Текст промпта или None, если не найден
        """
        return self.prompts.get(key)
    
    def get_all_prompts(self) -> Dict[str, str]:
        """
        Получает все загруженные промпты.
        
        Returns:
            Словарь со всеми промптами
        """
        return self.prompts.copy()
    
    def get_combined_prompt(self, separator: str = "\n\n---\n\n") -> str:
        """
        Объединяет все промпты в один текст.
        
        Args:
            separator: Разделитель между промптами
            
        Returns:
            Объединенный текст всех промптов
        """
        if not self.prompts:
            return ""
        
        parts = []
        for key, prompt in self.prompts.items():
            parts.append(f"=== {key.upper()} ===\n{prompt}")
        
        return separator.join(parts)
    
    def reload(self) -> Dict[str, str]:
        """
        Перезагружает промпты из файлов.
        
        Полезно, если файлы промптов были изменены во время работы.
        
        Returns:
            Словарь с обновленными промптами
        """
        logger.info("Перезагрузка промптов...")
        return self.load_prompts()