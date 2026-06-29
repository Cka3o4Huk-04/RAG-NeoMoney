"""
Модуль для синхронизации с GitHub.

Функционал:
1. Проверка изменений в удаленном репозитории
2. Скачивание обновлений для knowledge_base и prompts
3. Автоматическое обновление RAG системы
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class GitHubSync:
    """
    Класс для синхронизации с GitHub репозиторием.
    
    Проверяет изменения в knowledge_base и prompts,
    скачивает обновления и инициирует перезагрузку системы.
    """
    
    def __init__(
        self, 
        repo_url: str,
        knowledge_base_dir: str = "knowledge_base",
        prompts_dir: str = "prompts",
        temp_dir: str = "temp_sync"
    ):
        """
        Инициализация синхронизатора.
        
        Args:
            repo_url: URL GitHub репозитория
            knowledge_base_dir: Локальная директория базы знаний
            prompts_dir: Локальная директория промптов
            temp_dir: Временная директория для скачивания
        """
        self.repo_url = repo_url
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.prompts_dir = Path(prompts_dir)
        self.temp_dir = Path(temp_dir)
        
        logger.info(f"GitHubSync инициализирован (repo: {repo_url})")
    
    def check_for_updates(self) -> bool:
        """
        Проверяет наличие обновлений в удаленном репозитории.
        
        Returns:
            True если есть обновления, False иначе
        """
        try:
            # Создаем временную директорию
            self.temp_dir.mkdir(exist_ok=True)
            
            # Клонируем репозиторий во временную папку
            logger.info("Проверка обновлений в GitHub...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.temp_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка при клонировании: {result.stderr}")
                return False
            
            # Сравниваем directories
            kb_changed = self._compare_directories(
                self.knowledge_base_dir,
                self.temp_dir / "knowledge_base"
            )
            
            prompts_changed = self._compare_directories(
                self.prompts_dir,
                self.temp_dir / "prompts"
            )
            
            # Очищаем временную директорию
            self._cleanup_temp()
            
            has_updates = kb_changed or prompts_changed
            
            if has_updates:
                logger.info("Обнаружены обновления в GitHub!")
                if kb_changed:
                    logger.info("- Изменения в knowledge_base")
                if prompts_changed:
                    logger.info("- Изменения in prompts")
            else:
                logger.info("Обновлений не обнаружено")
            
            return has_updates
            
        except Exception as e:
            logger.error(f"Ошибка при проверке обновлений: {e}")
            return False
    def sync_from_github(self) -> Tuple[bool, bool]:
        """
        Синхронизирует локальные файлы с GitHub.
        
        Returns:
            Кортеж (knowledge_base_updated, prompts_updated)
        """
        try:
            logger.info("Начало синхронизации с GitHub...")
            
            # Создаем временную директорию
            self.temp_dir.mkdir(exist_ok=True)
            
            # Клонируем репозиторий
            logger.info("Скачивание обновлений из GitHub...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.temp_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка при клонировании: {result.stderr}")
                return False, False
            
            # Синхронизируем directories
            kb_updated = self._sync_directory(
                self.temp_dir / "knowledge_base",
                self.knowledge_base_dir
            )
            
            prompts_updated = self._sync_directory(
                self.temp_dir / "prompts",
                self.prompts_dir
            )
            
            # Очищаем временную директорию
            self._cleanup_temp()
            
            logger.info("Синхронизация завершена!")
            if kb_updated:
                logger.info("✅ knowledge_base обновлен")
            if prompts_updated:
                logger.info("✅ prompts обновлены")
            
            return kb_updated, prompts_updated
            
        except Exception as e:
            logger.error(f"Ошибка при синхронизации: {e}")
            return False, False
    
    def sync_with_rag_update(self, vector_store) -> Tuple[bool, bool]:
        """
        Синхронизирует файлы и обновляет RAG систему.
        
        Args:
            vector_store: Экземпляр VectorStore для обновления базы данных
            
        Returns:
            Кортеж (knowledge_base_updated, prompts_updated)
        """
        try:
            logger.info("Начало синхронизации с обновлением RAG...")
            
            # Синхронизируем файлы
            kb_updated, prompts_updated = self.sync_from_github()
            
            if kb_updated or prompts_updated:
                logger.info("Обнаружены изменения. Обновление RAG системы...")
                
                # Очищаем базу данных перед обновлением
                logger.info("Очистка базы данных ChromaDB...")
                vector_store.clear_collection()
                
                # Загружаем обновленные документы
                from document_loader import DocumentLoader
                
                doc_loader = DocumentLoader(knowledge_base_dir=str(self.knowledge_base_dir))
                documents = doc_loader.load_all_documents()
                
                if documents:
                    logger.info(f"Загрузка {len(documents)} документов в RAG систему...")
                    vector_store.add_documents(documents)
                    logger.info("✅ RAG система обновлена")
                else:
                    logger.warning("Не найдено документов для загрузки после синхронизации")
            
            return kb_updated, prompts_updated
            
        except Exception as e:
            logger.error(f"Ошибка при синхронизации с обновлением RAG: {e}")
            return False, False
    def sync_from_github(self) -> Tuple[bool, bool]:
        """
        Синхронизирует локальные файлы с GitHub.
        
        Returns:
            Кортеж (knowledge_base_updated, prompts_updated)
        """
        try:
            logger.info("Начало синхронизации с GitHub...")
            
            # Создаем временную директорию
            self.temp_dir.mkdir(exist_ok=True)
            
            # Клонируем репозиторий
            logger.info("Скачивание обновлений из GitHub...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.temp_dir)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка при клонировании: {result.stderr}")
                return False, False
            
            # Синхронизируем directories
            kb_updated = self._sync_directory(
                self.temp_dir / "knowledge_base",
                self.knowledge_base_dir
            )
            
            prompts_updated = self._sync_directory(
                self.temp_dir / "prompts",
                self.prompts_dir
            )
            
            # Очищаем временную директорию
            self._cleanup_temp()
            
            logger.info("Синхронизация завершена!")
            if kb_updated:
                logger.info("✅ knowledge_base обновлен")
            if prompts_updated:
                logger.info("✅ prompts обновлены")
            
            return kb_updated, prompts_updated
            
        except Exception as e:
            logger.error(f"Ошибка при синхронизации: {e}")
            return False, False
    
    def _compare_directories(self, local_dir: Path, remote_dir: Path) -> bool:
        """
        Сравнивает локальную и удаленную директории.
        """
        if not local_dir.exists() or not remote_dir.exists():
            return True
        
        local_files = list(local_dir.rglob("*"))
        remote_files = list(remote_dir.rglob("*"))
        
        if len(local_files) != len(remote_files):
            return True
        
        for remote_file in remote_files:
            if remote_file.is_file():
                relative_path = remote_file.relative_to(remote_dir)
                local_file = local_dir / relative_path
                
                if not local_file.exists():
                    return True
                
                if remote_file.stat().st_size != local_file.stat().st_size:
                    return True
        
        return False
    
    def _sync_directory(self, source: Path, destination: Path) -> bool:
        """
        Синхронизирует директорию.
        """
        if not source.exists():
            logger.warning(f"Исходная директория не существует: {source}")
            return False
        
        try:
            destination.mkdir(parents=True, exist_ok=True)
            
            for item in source.rglob("*"):
                relative_path = item.relative_to(source)
                dest_path = destination / relative_path
                
                if item.is_file():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
                    logger.debug(f"Синхронизирован файл: {relative_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при синхронизации директории: {e}")
            return False
    
    def _cleanup_temp(self):
        """Очищает временную директорию."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.debug("Временная директория очищена")
            except Exception as e:
                logger.warning(f"Ошибка при очистке временной директории: {e}")