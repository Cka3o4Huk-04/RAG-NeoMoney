"""
Модуль для логирования взаимодействий с ассистентом в базу данных.

Логирует все запросы пользователей и ответы ассистента для последующего анализа.
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Generator
from pathlib import Path
import csv
import io

logger = logging.getLogger(__name__)


class DatabaseLogger:
    """
    Класс для логирования взаимодействий в SQLite базу данных.
    
    Хранит:
    - Вопросы пользователей
    - Ответы ассистента
    - Метаданные (время, источник, user_id для Telegram)
    - Статус (из кеша или новый запрос)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Инициализация логгера базы данных.
        
        Args:
            db_path: Путь к файлу базы данных SQLite (если None, берется из env)
        """
        self.db_path = Path(db_path or os.getenv("LOGS_DB_FILE", "logs.db"))
        self._init_database()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Контекстный менеджер для получения соединения с базой данных."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Ошибка базы данных: {e}")
            raise
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Создает таблицу для логов, если она не существует."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        username TEXT,
                        source TEXT NOT NULL,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        from_cache INTEGER DEFAULT 0,
                        response_time_ms INTEGER,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Создаем индексы для быстрого поиска
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON logs(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON logs(source)")
                
                logger.debug(f"База данных инициализирована: {self.db_path}")
        except Exception as e:
            logger.error(f"Не удалось инициализировать базу данных: {e}")
            raise
    
    def log_interaction(
        self,
        query: str,
        response: str,
        source: str = "console",
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        from_cache: bool = False,
        response_time_ms: Optional[int] = None
    ) -> None:
        """
        Логирует взаимодействие пользователя с ассистентом.
        
        Args:
            query: Вопрос пользователя
            response: Ответ ассистента
            source: Источник запроса (console, telegram, api и т.д.)
            user_id: ID пользователя (для Telegram)
            username: Имя пользователя
            from_cache: Был ли ответ взят из кеша
            response_time_ms: Время ответа в миллисекундах
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                
                cursor.execute("""
                    INSERT INTO logs (
                        timestamp, user_id, username, source, query, response, 
                        from_cache, response_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    user_id,
                    username,
                    source,
                    query,
                    response,
                    1 if from_cache else 0,
                    response_time_ms
                ))
                
                logger.debug(f"Запись добавлена в логи: source={source}, from_cache={from_cache}")
        except Exception as e:
            logger.error(f"Не удалось записать лог: {e}")
    
    def get_logs(
        self,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> list:
        """
        Получает логи из базы данных с фильтрацией.
        
        Args:
            limit: Максимальное количество записей
            user_id: Фильтр по ID пользователя
            source: Фильтр по источнику
            start_date: Начальная дата (ISO format)
            end_date: Конечная дата (ISO format)
            
        Returns:
            Список словарей с логами
        """
        try:
            with self.get_connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Строим запрос с параметрами
                where_clauses = []
                params = []
                
                if user_id:
                    where_clauses.append("user_id = ?")
                    params.append(user_id)
                
                if source:
                    where_clauses.append("source = ?")
                    params.append(source)
                
                if start_date:
                    where_clauses.append("timestamp >= ?")
                    params.append(start_date)
                
                if end_date:
                    where_clauses.append("timestamp <= ?")
                    params.append(end_date)
                
                where_sql = " AND ".join(where_clauses)
                if where_sql:
                    where_sql = " WHERE " + where_sql
                
                query = f"SELECT * FROM logs{where_sql} ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Преобразуем Row объекты в словари
                logs = [dict(row) for row in rows]
                
                logger.debug(f"Получено {len(logs)} записей из логов")
                return logs
        except Exception as e:
            logger.error(f"Не удалось получить логи: {e}")
            return []
    
    def export_to_csv(
        self,
        output_path: Optional[str] = None,
        user_id: Optional[str] = None,
        source: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """
        Экспортирует логи в CSV файл.
        
        Args:
            output_path: Путь к выходному файлу (если None, возвращает строку)
            user_id: Фильтр по ID пользователя
            source: Фильтр по источнику
            start_date: Начальная дата
            end_date: Конечная дата
            
        Returns:
            Путь к созданному файлу или содержимое CSV как строка
        """
        logs = self.get_logs(
            user_id=user_id,
            source=source,
            start_date=start_date,
            end_date=end_date
        )
        
        if not logs:
            logger.info("Нет данных для экспорта в CSV")
            return ""
        
        fieldnames = [
            'id', 'timestamp', 'user_id', 'username', 'source', 
            'query', 'response', 'from_cache', 'response_time_ms', 'created_at'
        ]
        
        if output_path:
            try:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(logs)
                logger.info(f"Логи экспортированы в CSV: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Не удалось экспортировать в CSV: {e}")
                return ""
        else:
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
            return output.getvalue()
    
    def get_stats(self) -> dict:
        """
        Получает статистику по логам.
        
        Returns:
            Словарь со статистикой
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Общее количество запросов
                cursor.execute("SELECT COUNT(*) FROM logs")
                total_requests = cursor.fetchone()[0]
                
                # Запросов из кеша
                cursor.execute("SELECT COUNT(*) FROM logs WHERE from_cache = 1")
                cached_requests = cursor.fetchone()[0]
                
                # Уникальных пользователей
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM logs WHERE user_id IS NOT NULL")
                unique_users = cursor.fetchone()[0]
                
                # По источникам
                cursor.execute("SELECT source, COUNT(*) FROM logs GROUP BY source")
                by_source = dict(cursor.fetchall())
                
                # Среднее время ответа
                cursor.execute("SELECT AVG(response_time_ms) FROM logs WHERE response_time_ms IS NOT NULL")
                avg_response_time = cursor.fetchone()[0]
                
                stats = {
                    'total_requests': total_requests or 0,
                    'cached_requests': cached_requests or 0,
                    'unique_users': unique_users or 0,
                    'by_source': by_source or {},
                    'avg_response_time_ms': avg_response_time or 0
                }
                
                logger.debug(f"Статистика получена: {stats}")
                return stats
        except Exception as e:
            logger.error(f"Не удалось получить статистику: {e}")
            return {
                'total_requests': 0,
                'cached_requests': 0,
                'unique_users': 0,
                'by_source': {},
                'avg_response_time_ms': 0
            }
    
    def close(self) -> None:
        """Закрывает соединение с базой данных (если нужно)."""
        # В текущей реализации с context manager это не требуется
        pass
