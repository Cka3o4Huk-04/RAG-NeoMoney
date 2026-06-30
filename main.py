"""
Главный файл для запуска RAG-ассистента.

Это точка входа в приложение. Здесь происходит:
1. Загрузка конфигурации
2. Инициализация всех компонентов (кеш, векторная база, RAG)
3. Добавление примеров документов (при первом запуске)
4. Интерактивный цикл общения с пользователем
"""

import logging
import os
import time
from typing import Optional
from dotenv import load_dotenv
from embeddings import VectorStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache
from db_logger import DatabaseLogger
from telegram_bot import TelegramRAGBot
from prompt_loader import PromptLoader
from document_loader import DocumentLoader
from github_sync import GitHubSync
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

# Настраиваем базовое логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_assistant.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Глобальные настройки RAG (считываются из .env)
rag_top_k: int = 3
min_relevance_score: float = 0.5
chunk_size: int = 500
chunk_overlap: int = 50


def initialize_system():
    """
    Инициализирует все компоненты RAG-системы.
    
    Returns:
        Кортеж (embedding_store, rag_assistant, cache, logger)
    """
    logger.info("=" * 70)
    logger.info("🚀 ИНИЦИАЛИЗАЦИЯ RAG-АССИСТЕНТА")
    logger.info("=" * 70)
    
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Определяем провайдера API (OpenAI или ProxyAPI)
    use_proxyapi = os.getenv("USE_PROXYAPI", "false").lower() == "true"
    
    if use_proxyapi:
        # Режим ProxyAPI
        api_key = os.getenv("PROXYAPI_KEY")
        base_url = os.getenv("PROXYAPI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
        provider_name = "ProxyAPI"
        
        if not api_key:
            logger.warning("Не найден PROXYAPI_KEY в переменных окружения!")
            logger.warning("Создайте файл .env и добавьте туда: PROXYAPI_KEY=your_key_here")
            logger.warning("Или установите USE_PROXYAPI=false для использования OpenAI API.")
    else:
        # Режим OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
        provider_name = "OpenAI"
        
        if not api_key:
            logger.warning("Не найден OPENAI_API_KEY в переменных окружения!")
            logger.warning("Создайте файл .env и добавьте туда: OPENAI_API_KEY=your_key_here")
            logger.warning("Или установите USE_PROXYAPI=true для использования ProxyAPI.")
    
    logger.info(f"📡 Провайдер API: {provider_name}")
    
    # Получаем настройки моделей из переменных окружения
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    temperature = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Настройки качества RAG (обновляем глобальные переменные)
    global rag_top_k, min_relevance_score, chunk_size, chunk_overlap
    rag_top_k = int(os.getenv("RAG_TOP_K", "3"))
    min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.5"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    logger.info(f"Настройки RAG: top_k={rag_top_k}, min_relevance={min_relevance_score}, chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    # 1. Инициализируем кеш для хранения ответов
    logger.info("[1/5] Инициализация кеша...")
    cache_file = os.getenv("CACHE_FILE", "cache.json")
    cache = ResponseCache(cache_file=cache_file)
    
    # 1.1. Загружаем системные промпты
    logger.info("[2/5] Загрузка системных промптов...")
    prompts_dir = os.getenv("PROMPTS_DIR", "prompts")
    prompt_loader = PromptLoader(prompts_dir=prompts_dir)
    system_prompt = prompt_loader.get_combined_prompt()
    if system_prompt:
        logger.info(f"✓ Загружено системных промптов: {len(prompt_loader.get_all_prompts())}")
    else:
        logger.warning("Системные промпты не найдены, используется промпт по умолчанию")
    
    # 2. Инициализируем векторное хранилище ChromaDB
    logger.info("[3/5] Инициализация векторного хранилища...")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    vector_store = VectorStore(
        collection_name="rag_documents",
        persist_directory=chroma_dir,
        embedding_model=embedding_model,
        api_key=api_key,
        base_url=base_url
    )
    
    # 2.1. Загружаем документы из knowledge_base
    logger.info("[3.1/5] Загрузка документов из базы знаний...")
    knowledge_base_dir = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    document_loader = DocumentLoader(knowledge_base_dir=knowledge_base_dir)
    documents = document_loader.load_all_documents()
    
    if documents:
        logger.info(f"✓ Загружено документов из knowledge_base: {len(documents)}")
        # Добавляем документы в векторное хранилище
        vector_store.add_documents(documents)
    else:
        logger.info("Документы в knowledge_base не найдены")
    
    # Проверяем, нужно ли добавить примеры документов
    if vector_store.collection.count() == 0:
        logger.info("База данных пуста. Добавляем примеры документов...")
        sample_docs = get_sample_documents()
        vector_store.add_documents(sample_docs)
    else:
        logger.info(f"✓ В базе уже есть {vector_store.collection.count()} документов")
    
    # 3. Инициализируем RAG-ассистента с системным промптом
    logger.info("[4/5] Инициализация RAG-ассистента...")
    rag_assistant = RAGAssistant(
        vector_store=vector_store,
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        temperature=temperature,
        system_prompt=system_prompt  # Передаем загруженный системный промпт
    )
    
    # 4. Инициализируем логгер базы данных
    logger.info("[5/5] Инициализация логгера базы данных...")
    logs_db = os.getenv("LOGS_DB_FILE", "logs.db")
    db_logger = DatabaseLogger(db_path=logs_db)
    logger.info("✓ Логгер инициализирован")
    
    logger.info("=" * 70)
    logger.info("✅ СИСТЕМА ГОТОВА К РАБОТЕ")
    logger.info("=" * 70)
    
    return vector_store, rag_assistant, cache, db_logger


def answer_question(
    query: str,
    rag_assistant: RAGAssistant,
    cache: ResponseCache,
    db_logger: DatabaseLogger,
    source: str = "console",
    user_id: Optional[str] = None,
    username: Optional[str] = None
) -> str:
    """
    Отвечает на вопрос пользователя с использованием кеша и RAG.
    
    Логика работы:
    1. Проверяем кеш - если ответ есть, возвращаем его
    2. Если ответа нет, выполняем RAG (поиск + генерация)
    3. Сохраняем новый ответ в кеш
    4. Логируем взаимодействие в базу данных
    5. Возвращаем ответ
    
    Args:
        query: Вопрос пользователя
        rag_assistant: Экземпляр RAG-ассистента
        cache: Экземпляр кеша
        db_logger: Экземпляр логгера базы данных
        source: Источник запроса (console, telegram и т.д.)
        user_id: ID пользователя (для Telegram)
        username: Имя пользователя
        
    Returns:
        Ответ на вопрос
    """
    logger.info("=" * 70)
    logger.info(f"❓ ВОПРОС: {query}")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Шаг 1: Проверяем кеш
    logger.info("[Шаг 1] Проверка кеша...")
    cached_answer = cache.get(query)
    from_cache = cached_answer is not None
    
    if cached_answer:
        # Ответ найден в кеше - возвращаем его
        logger.info("💾 Ответ из кеша:")
        logger.info("-" * 70)
        logger.info(cached_answer)
        logger.info("-" * 70)
        answer = cached_answer
    else:
        # Шаг 2: Ответа нет в кеше - выполняем RAG
        logger.info("[Шаг 2] Выполнение RAG (поиск + генерация)...")
        
        try:
            answer, search_results = rag_assistant.generate_response(
                query=query,
                top_k=rag_top_k,
                verbose=True
            )
            
            # Шаг 3: Сохраняем ответ в кеш
            logger.info("[Шаг 3] Сохранение ответа в кеш...")
            cache.set(query, answer)
            
            # Выводим финальный ответ
            logger.info("💡 ОТВЕТ:")
            logger.info("-" * 70)
            logger.info(answer)
            logger.info("-" * 70)
            
        except Exception as e:
            error_msg = f"Ошибка при обработке запроса: {str(e)}"
            logger.error(f"❌ {error_msg}")
            answer = error_msg
    
    # Шаг 4: Логируем взаимодействие
    response_time_ms = int((time.time() - start_time) * 1000)
    db_logger.log_interaction(
        query=query,
        response=answer,
        source=source,
        user_id=user_id,
        username=username,
        from_cache=from_cache,
        response_time_ms=response_time_ms
    )
    
    return answer


def sync_with_github(vector_store, rag_assistant):
    """
    Синхронизация с GitHub и обновление RAG системы.
    """
    print("\n" + "=" * 70)
    print("🔄 СИНХРОНИЗАЦИЯ С GITHUB")
    print("=" * 70)
    
    repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/Cka3o4Huk-04/RAG-NeoMoney")
    
    syncer = GitHubSync(repo_url=repo_url)
    
    # Проверяем наличие обновлений
    print("\nПроверка обновлений...")
    has_updates = syncer.check_for_updates()
    
    if has_updates:
        print("\n⚠️ Обнаружены обновления в репозитории!")
        confirm = input("Скачать и применить обновления? (y/n): ").strip().lower()
        if confirm in ['y', 'yes', 'д', 'да']:
            print("\nНачало синхронизации...")
            kb_updated, prompts_updated = syncer.sync_with_rag_update(vector_store)
            
            if kb_updated or prompts_updated:
                print("\n✅ Синхронизация завершена!")
                if kb_updated:
                    print("  • База знаний обновлена")
                if prompts_updated:
                    print("  • Промпты обновлены")
            else:
                print("\n⚠️ Синхронизация не удалась")
        else:
            print("\nСинхронизация отменена пользователем")
    else:
        print("\n✅ Обновлений не обнаружено. Система актуальна.")
    
    print("=" * 70)


def auto_sync_job(vector_store, rag_assistant, db_logger):
    """
    Задача для автоматической синхронизации с GitHub.
    Вызывается планировщиком в заданное time.
    """
    logger.info("=" * 70)
    logger.info("🔄 АВТОМАТИЧЕСКАЯ СИНХРОНИЗАЦИЯ (3:00)")
    logger.info("=" * 70)
    
    repo_url = os.getenv("GITHUB_REPO_URL", "https://github.com/Cka3o4Huk-04/RAG-NeoMoney")
    admin_id = os.getenv("TELEGRAM_ADMIN_ID")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    syncer = GitHubSync(repo_url=repo_url)
    
    try:
        # Проверяем наличие обновлений
        logger.info("Проверка обновлений...")
        has_updates = syncer.check_for_updates()
        
        if has_updates:
            logger.info("⚠️ Обнаружены обновления в репозитории!")
            
            # Выполняем синхронизацию
            kb_updated, prompts_updated = syncer.sync_with_rag_update(vector_store)
            
            if kb_updated or prompts_updated:
                logger.info("✅ Синхронизация завершена!")
                if kb_updated:
                    logger.info("  • База знаний обновлена")
                if prompts_updated:
                    logger.info("  • Промпты обновлены")
                
                # Отправляем уведомление администратору
                if admin_id and bot_token:
                    try:
                        send_telegram_notification(bot_token, admin_id, 
                            "✅ Автоматическая синхронизация завершена!\n\n"
                            + ("База знаний обновлена\n" if kb_updated else "")
                            + ("Промпты обновлены\n" if prompts_updated else "")
                        )
                    except Exception as e:
                        logger.warning(f"Не удалось отправить уведомление: {e}")
            else:
                logger.warning("⚠️ Синхронизация не удалась")
                if admin_id and bot_token:
                    try:
                        send_telegram_notification(bot_token, admin_id,
                            "⚠️ Автоматическая синхронизация не удалась!"
                        )
                    except Exception as e:
                        logger.warning(f"Не удалось отправить уведомление: {e}")
        else:
            logger.info("✅ Обновлений не обнаружено. Система актуальна.")
            
    except Exception as e:
        logger.error(f"❌ Ошибка при автоматической синхронизации: {e}", exc_info=True)
        if admin_id and bot_token:
            try:
                send_telegram_notification(bot_token, admin_id,
                    f"❌ Ошибка автоматической синхронизации: {str(e)}"
                )
            except Exception as e2:
                logger.warning(f"Не удалось отправить уведомление: {e2}")


def send_telegram_notification(bot_token, chat_id, message):
    """
    Отправляет уведомление в Telegram.
    """
    import asyncio
    from telegram import Bot
    
    async def send():
        bot = Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message)
    
    asyncio.run(send())
    logger.info(f"Уведомление отправлено в Telegram (chat_id: {chat_id})")
def setup_auto_sync(vector_store, rag_assistant, db_logger):
    """
    Настраивает автоматическую синхронизацию по расписанию.
    """
    auto_sync_enabled = os.getenv("AUTO_SYNC_ENABLED", "false").lower() == "true"
    auto_sync_time = os.getenv("AUTO_SYNC_TIME", "03:00")
    
    if not auto_sync_enabled:
        logger.info("Автоматическая синхронизация отключена (AUTO_SYNC_ENABLED=false)")
        return None
    
    try:
        # Парсим время (формат ЧЧ:ММ)
        hour, minute = map(int, auto_sync_time.split(':'))
        
        # Создаем планировщик
        scheduler = AsyncIOScheduler()
        
        # Добавляем задачу на ежедневное выполнение в заданное time
        scheduler.add_job(
            auto_sync_job,
            'cron',
            hour=hour,
            minute=minute,
            args=[vector_store, rag_assistant, db_logger],
            id='auto_sync_github',
            name='Автоматическая синхронизация с GitHub',
            replace_existing=True
        )
        
        # Запускаем планировщик
        scheduler.start()
        
        logger.info(f"✅ Автоматическая синхронизация настроена на {auto_sync_time}")
        logger.info(f"   Следующий запуск: {scheduler.get_job('auto_sync_github').next_run_time}")
        
        return scheduler
        
    except Exception as e:
        logger.error(f"❌ Ошибка при настройке автоматической синхронизации: {e}")
        return None


def interactive_mode(rag_assistant: RAGAssistant, cache: ResponseCache, logger: DatabaseLogger, vector_store=None):
    """
    Интерактивный режим общения с ассистентом.
    
    Пользователь может задавать вопросы в цикле до тех пор,
    пока не введет команду выхода.
    """
    print("\n" + "=" * 70)
    print("💬 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nВы можете задавать вопросы ассистенту.")
    print("Для выхода введите: exit, quit, выход или q")
    print("\nДоступные команды:")
    print("  • cache - показать информацию о кеше")
    print("  • clear_cache - очистить кеш")
    print("  • stats - показать статистику системы")
    print("  • logs - экспортировать логи в CSV")
    print()
    
    while True:
        try:
            # Получаем ввод от пользователя
            user_input = input("\n👤 Вы: ").strip()
            
            # Проверяем команды выхода
            if user_input.lower() in ['exit', 'quit', 'выход', 'q', '']:
                print("\n👋 До свидания!")
                break
            
            # Обрабатываем специальные команды
            if user_input.lower() == 'cache':
                print(f"\n📊 Кеш содержит {cache.size()} записей")
                continue
            
            if user_input.lower() == 'clear_cache':
                cache.clear()
                print("\n✓ Кеш очищен")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n📊 СТАТИСТИКА СИСТЕМЫ:")
                print(f"  • Документов в ChromaDB: {rag_assistant.vector_store.collection.count()}")
                print(f"  • Записей в кеше: {cache.size()}")
                print(f"  • Модель LLM: {rag_assistant.model}")
                
                # Показываем статистику из логов
                log_stats = logger.get_stats()
                print(f"\n📝 ЛОГИ:")
                print(f"  • Всего запросов: {log_stats['total_requests']}")
                print(f"  • Из кеша: {log_stats['cached_requests']}")
                print(f"  • Среднее время ответа: {log_stats['avg_response_time_ms']:.0f} мс")
                continue
            
            if user_input.lower() == 'logs':
                filename = f"logs_console_{int(time.time())}.csv"
                logger.export_to_csv(output_path=filename, source="console")
                print(f"\n✓ Логи экспортированы в файл: {filename}")
                continue
            
            # Обрабатываем вопрос пользователя
            answer_question(user_input, rag_assistant, cache, logger, source="console")
            
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")


def demo_mode(rag_assistant: RAGAssistant, cache: ResponseCache, logger: DatabaseLogger):
    """
    Демонстрационный режим с заранее заготовленными вопросами.
    
    Показывает работу системы на примерах, включая использование кеша.
    """
    print("\n" + "=" * 70)
    print("🎬 ДЕМОНСТРАЦИОННЫЙ РЕЖИМ")
    print("=" * 70)
    print("\nСейчас будет продемонстрирована работа RAG-ассистента")
    print("на нескольких примерах вопросов.\n")
    
    # Список демо-вопросов
    demo_questions = [
        "Что такое Python и для чего он используется?",
        "Расскажи про RAG и как он работает",
        "Что такое векторные базы данных?",
        "Что такое Python и для чего он используется?"  # Повторный вопрос для демонстрации кеша
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#' * 70}")
        print(f"ВОПРОС {i} из {len(demo_questions)}")
        print(f"{'#' * 70}")
        
        answer_question(question, rag_assistant, cache, logger, source="console")
        
        # Пауза между вопросами (кроме последнего)
        if i < len(demo_questions):
            input("\n[Нажмите Enter для следующего вопроса...]")
    
    print("\n\n" + "=" * 70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 70)


def main():
    """
    Главная функция приложения.
    """
    try:
        # Инициализируем систему
        vector_store, rag_assistant, cache, db_logger = initialize_system()
        
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        
        # Выбор режима работы
        print("\n" + "=" * 70)
        print("ВЫБОР РЕЖИМА РАБОТЫ")
        print("=" * 70)
        print("\n1. Интерактивный режим - задавайте свои вопросы")
        print("2. Демонстрационный режим - готовые примеры вопросов")
        if telegram_token:
            print("3. Telegram бот - запуск бота для Telegram")
        print("4. Синхронизация с GitHub - обновить базу знаний и промпты")
        print()
        
        mode = input("Выберите режим (1, 2" + (", 3" if telegram_token else "") + ", 4, по умолчанию 1): ").strip()
        
        if mode == '2':
            demo_mode(rag_assistant, cache, db_logger)
            
            # Предложить перейти в интерактивный режим
            print("\n" + "=" * 70)
            continue_interactive = input("\nПерейти в интерактивный режим? (y/n): ").strip().lower()
            if continue_interactive in ['y', 'yes', 'д', 'да', '']:
                interactive_mode(rag_assistant, cache, db_logger)
        elif mode == '3' and telegram_token:
            # Запускаем Telegram бота
            print("\n" + "=" * 70)
            print("🤖 ЗАПУСК TELEGRAM БОТА")
            print("=" * 70)
            bot = TelegramRAGBot(
                token=telegram_token,
                rag_assistant=rag_assistant,
                cache=cache,
                logger=db_logger
            )
            bot.run()
        elif mode == '4':
            sync_with_github(vector_store, rag_assistant)
        else:
            # Настраиваем автоматическую синхронизацию
            scheduler = setup_auto_sync(vector_store, rag_assistant, db_logger)
            
            try:
                interactive_mode(rag_assistant, cache, db_logger, vector_store)
            finally:
                # Останавливаем планировщик при выходе
                if scheduler:
                    scheduler.shutdown()
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()

