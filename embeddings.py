"""
Модуль для работы с векторным хранилищем ChromaDB и эмбеддингами.

Здесь мы создаем векторные представления текстов используя OpenAI API
и сохраняем их в ChromaDB для быстрого семантического поиска.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Класс для работы с векторным хранилищем ChromaDB.
    
    Использует OpenAI API (или ProxyAPI) для создания эмбеддингов
    и ChromaDB для их хранения и поиска.
    """
    
    def __init__(
        self, 
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: Имя коллекции в ChromaDB
            persist_directory: Директория для сохранения данных ChromaDB
            embedding_model: Название модели OpenAI для эмбеддингов
            api_key: API ключ OpenAI/ProxyAPI (если None, берется из переменной окружения)
            base_url: Базовый URL API (для ProxyAPI: https://api.proxyapi.ru/openai/v1)
        """
        logger.info(f"Инициализация ChromaDB в директории: {persist_directory}")
        
        # Создаем клиент ChromaDB с персистентным хранилищем
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False  # Отключаем телеметрию
            )
        )
        
        # Инициализируем клиент OpenAI для создания эмбеддингов
        client_kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if base_url:
            client_kwargs["base_url"] = base_url
            logger.info(f"Используется кастомный API endpoint: {base_url}")
        
        self.openai_client = OpenAI(**client_kwargs)
        self.embedding_model = embedding_model
        
        logger.info(f"Модель эмбеддингов: {embedding_model} (OpenAI API)")
        
        # Получаем или создаем коллекцию в ChromaDB
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Документы для RAG-ассистента"}
        )
        
        logger.info(f"✓ ChromaDB инициализирована. Документов в коллекции: {self.collection.count()}")
    
    def _create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Разбивает текст на чанки (фрагменты) с перекрытием.
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка в символах
            overlap: Размер перекрытия между чанками
            
        Returns:
            Список чанков текста
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для списка текстов используя OpenAI API.
        
        Args:
            texts: Список текстов для создания эмбеддингов
            
        Returns:
            Список векторов эмбеддингов
        """
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                encoding_format="float"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise
    
    def add_documents(self, documents: List[Tuple[str, str]]) -> None:
        """
        Добавляет документы в векторное хранилище.
        
        Args:
            documents: Список кортежей (название_документа, текст_документа)
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        chunk_id = self.collection.count()
        
        logger.info(f"Добавление {len(documents)} документов в ChromaDB...")
        
        for doc_name, doc_text in documents:
            chunks = self._create_chunks(doc_text)
            logger.info(f"  • {doc_name}: {len(chunks)} чанков")
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": doc_name,
                    "chunk_length": len(chunk)
                })
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        
        logger.info(f"Создание эмбеддингов для {len(all_chunks)} чанков через OpenAI API...")
        logger.info(f"(Модель: {self.embedding_model})")
        
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            logger.info(f"  Обработка чанков {i+1}-{min(i+batch_size, len(all_chunks))} из {len(all_chunks)}...")
            batch_embeddings = self._create_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
        
        logger.info("Сохранение в ChromaDB...")
        self.collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        logger.info(f"✓ Добавлено {len(all_chunks)} чанков. Всего в базе: {self.collection.count()}")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Выполняет семантический поиск по векторному хранилищу.
        
        Args:
            query: Поисковый запрос пользователя
            top_k: Количество результатов для возврата
            
        Returns:
            Список кортежей (текст_чанка, источник, расстояние)
        """
        if self.collection.count() == 0:
            logger.warning("Коллекция пуста, нет документов для поиска")
            return []
        
        query_embeddings = self._create_embeddings([query])
        query_embedding = query_embeddings[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                chunk_text = results['documents'][0][i]
                source = results['metadatas'][0][i]['source']
                distance = results['distances'][0][i]
                formatted_results.append((chunk_text, source, distance))
        
        return formatted_results
    
    def clear_collection(self) -> None:
        """Очищает коллекцию (удаляет все документы)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Документы для RAG-ассистента"}
        )
        logger.info("✓ Коллекция очищена")


def load_documents_from_folder(folder_path: str = "docs") -> List[Tuple[str, str]]:
    """
    Загружает документы из папки с txt файлами.
    
    Args:
        folder_path: Путь к папке с документами
        
    Returns:
        Список кортежей (название_файла, текст_документа)
    """
    docs_path = Path(folder_path)
    documents = []
    
    if not docs_path.exists():
        logger.warning(f"Папка {folder_path} не найдена")
        return documents
    
    if not docs_path.is_dir():
        logger.warning(f"{folder_path} не является папкой")
        return documents
    
    txt_files = list(docs_path.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"В папке {folder_path} не найдено .txt файлов")
        return documents
    
    logger.info(f"📂 Найдено {len(txt_files)} файлов в папке {folder_path}")
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content:
                doc_name = txt_file.stem
                documents.append((doc_name, content))
                logger.info(f"  ✓ Загружен: {txt_file.name}")
            else:
                logger.warning(f"  ⚠ Пропущен (пустой): {txt_file.name}")
                
        except Exception as e:
            logger.error(f"Ошибка при чтении {txt_file.name}: {e}")
    
    return documents


def get_sample_documents() -> List[Tuple[str, str]]:
    """
    Возвращает примеры документов для демонстрации RAG.
    
    Сначала пытается загрузить документы из папки docs/,
    если папка пуста или не существует, возвращает встроенные примеры.
    
    Returns:
        Список кортежей (название, текст)
    """
    documents = load_documents_from_folder("docs")
    
    if not documents:
        logger.info("📝 Используются встроенные примеры документов")
        documents = [
            (
                "Python Основы",
                """Python - это высокоуровневый язык программирования общего назначения. 
                Он был создан Гвидо ван Россумом и впервые выпущен в 1991 году.
                
                Python известен своей простотой и читаемостью кода. Философия языка 
                подчеркивает важность читаемости кода и позволяет программистам 
                выражать концепции в меньшем количестве строк кода, чем это было бы 
                возможно в других языках.
                
                Основные возможности Python включают:
                - Динамическую типизацию
                - Автоматическое управление памятью
                - Обширную стандартную библиотеку
                - Поддержку множественных парадигм программирования
                
                Python широко используется в веб-разработке, анализе данных, 
                машинном обучении, автоматизации и научных вычислениях."""
            ),
            (
                "Машинное обучение и AI",
                """Машинное обучение (Machine Learning) - это подраздел искусственного 
                интеллекта, который изучает алгоритмы и статистические модели, 
                позволяющие компьютерам выполнять задачи без явного программирования.
                
                Основные типы машинного обучения:
                
                1. Обучение с учителем (Supervised Learning)
                В этом подходе модель обучается на размеченных данных, где каждый 
                пример имеет известный правильный ответ. Примеры: классификация 
                изображений, предсказание цен на недвижимость.
                
                2. Обучение без учителя (Unsupervised Learning)
                Модель ищет закономерности в неразмеченных данных. Примеры: 
                кластеризация клиентов, обнаружение аномалий.
                
                3. Обучение с подкреплением (Reinforcement Learning)
                Агент обучается принимать решения, взаимодействуя со средой и 
                получая награды или штрафы.
                
                RAG (Retrieval-Augmented Generation) - это техника, которая улучшает 
                качество ответов языковых моделей, дополняя их внешними знаниями из 
                базы данных. Это позволяет модели давать более точные и актуальные 
                ответы, основанные на конкретных документах."""
            ),
            (
                "Векторные базы данных",
                """Векторные базы данных - это специализированные системы хранения данных, 
                оптимизированные для хранения и поиска векторных эмбеддингов.
                
                Что такое эмбеддинги?
                Эмбеддинги - это векторные представления данных (текста, изображений, 
                аудио) в многомерном пространстве. Семантически похожие объекты 
                располагаются близко друг к другу в этом пространстве.
                
                ChromaDB - это открытая векторная база данных, разработанная специально 
                для работы с эмбеддингами в приложениях с искусственным интеллектом.
                
                Преимущества ChromaDB:
                - Простота использования и встраивания в приложения
                - Поддержка персистентного хранения данных
                - Встроенная поддержка различных моделей эмбеддингов
                - Быстрый семантический поиск
                - Возможность работы как локально, так и в клиент-серверном режиме
                
                Векторные базы данных критически важны для RAG-систем, так как они 
                позволяют быстро находить релевантные документы на основе семантического 
                сходства запроса с содержимым базы данных.
                
                OpenAI предоставляет мощные модели для создания эмбеддингов, такие как 
                text-embedding-3-small и text-embedding-3-large. Эти модели создают 
                высококачественные векторные представления текста, которые отлично 
                работают для семантического поиска в различных языках, включая русский."""
            )
        ]
    
    return documents
