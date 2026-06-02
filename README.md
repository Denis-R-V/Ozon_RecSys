# Рекомендательная система fashion‑товаров на Ozon

## Команда проекта:
- участники:
    - Денис Рябинин
    - Дарья Клоконос
    - Виталий Минькевич
- куратор проекта:
    - Александра Ковалева

### Распределение ролей:
Каждый участник команды участвует в реализации каждого этапа и далее выбирается лучшее решение

## Цель проекта
В рамках проекта необходимо разработать масштабируемую систему рекомендаций типа "user-to-item", способную обрабатывать большой объем разнородных данных и формировать для каждого пользователя индивидуальный список из 100 товаров категории Apparel (одежда, обувь, аксессуары), которые с наибольшей вероятностью будут куплены (оформлен заказ).

## Метрика
- NDCG@100 на тестовой выборке (ml_ozon_recsys_test.snappy.parquet)

## Данные для обучения
Данные для обучения предоставлены Ozon и включают несколько источников:
- История взаимодействий пользователей с товарами (трекер событий): ~1,6 млрд событий за 6 месяцев (просмотры товаров, добавления в корзину, добавления в избранное, покупки и др.).
- История заказов: ~19 млн записей о заказах за тот же период с указанием финального статуса каждого заказа (доставлен, отменён, в процессе и т.п.).
- Каталог товаров: ~6,4 млн товаров категории Apparel с их атрибутами (текстовые, числовые, категориальные признаки) и готовыми эмбеддингами изображений товаров (512-мерные CLIP-эмбеддинги, специально дообученные под fashion-сегмент).
- Иерархия категорий товаров: дерево из ~7 тыс. категорий, к которым относятся товары.
- Дополнительные данные: предоставлен вспомогательный датасет (extra_ml_ozon_recsys_train.zip)

## План работы
1. Постановка задачи и планирование
2. Сбор данных и EDA
3. Разработка бейзлайна без использования ML
4. Разработка линейных моделей ML
5. Разработка нелинейных моделей ML, feature-engineering и выбор лучшего ML-решения
6. Создание сервиса с имплементацией лучшего ML-решения
7. Разработка DL-модели
8. Доработка задачи: улучшение сервисной части по обратной связи от команды курса П.Р.; тюнинг DL-моделей и выбор лучшего решения overall

## Структура ноутбуков

| Ноутбук | Описание |
|---------|----------|
| `notebooks/1_EDA.ipynb` | Разведочный анализ данных |
| `notebooks/2_datamart.ipynb` | Построение datamart / feature store |
| `notebooks/3_retrieval.ipynb` | Кандидатогенераторы (TopPopular, ALS, repeat-history, i2i) |
| `notebooks/4_DL_*.ipynb` | DL-модели (BERT4Rec, DSSM, SASRec) + merge |
| `notebooks/5_reranker.ipynb` | **Финальный реранкер** (2-stage: retrieval → LightGBM ranker) + MLflow PRD run + baseline comparison |
| `notebooks/6_prd_inference.ipynb` | Загрузка PRD-модели из MLflow, тестовый predict |

## Результаты финальной модели

Пул кандидатов: `merged_with_dl_all_repeat_history`, 512 шардов.  
Модель: LightGBM LambdaRank (`lgbm_ranker_t02_selection`).

| Разбивка | NDCG@100 | recall@100 | Примечание |
|----------|----------|-----------|-----------|
| TopPopular (baseline) | 0.005056 | 0.017450 | val_eval |
| ALS (baseline) | 0.004486 | 0.015342 | val_eval |
| Candidate raw rank (baseline) | 0.029241 | 0.097569 | val_eval |
| **2-stage reranker — val_eval** | **0.087751** | **0.145960** | финальная модель, val holdout |
| **2-stage reranker — test holdout** | **0.088141** | **0.151671** | Jul 2–15, честный тест |

Улучшение vs лучший baseline: **×17x** по NDCG@100.  
Ratio test/val = **1.00×** — модель обобщается без деградации.  
PRD run_id сохранён в `artifacts/reports/.../prd_run_id.txt`.

## Финальный статус чекпойнта

### Сравнение методов (val_eval, одинаковый пул кандидатов)

> График и таблица сохранены в MLflow artifact `reports/model_comparison.png`
> и `reports/model_comparison.csv` (PRD run `6e08071b`).

| Метод | NDCG@100 | Recall@100 | ×lift vs TopPop | ×lift vs best baseline |
|-------|----------|-----------|-----------------|------------------------|
| TopPopular | 0.005056 | 0.017450 | 1.0× | — |
| ALS | 0.004486 | 0.015342 | 0.9× | — |
| Candidate raw rank | 0.029241 | 0.097569 | 5.8× | 1.0× |
| **2-stage reranker (PRD)** | **0.087751** | **0.145960** | **17.4×** | **3.0×** |

### Параметры чекпойнта

| Параметр | Значение |
|----------|----------|
| **MLflow UI** | http://localhost:5000 |
| **MinIO UI** | http://localhost:9001 (minioadmin / minioadmin) |
| **PRD run_id** | `6e08071b6021407a8ed6c9dd28318a8b` |
| **Финальная модель** | 2-stage: retrieval (ALS + TopPop + DL + repeat-hist) → LightGBM ranker |
| **Inference notebook** | `notebooks/6_prd_inference.ipynb` — выполнен, модель из MLflow/MinIO artifact |

### Что залоггировано в PRD run

| Путь в MLflow | Описание |
|---------------|----------|
| `model/lgbm_ranker_t02_selection.joblib` | Обученная модель LightGBM |
| `model/feature_cols.json` | Список фич (240 numeric) |
| `notebooks/6_prd_inference.ipynb` | Выполненный inference-ноутбук |
| `reports/final_reranker_best_val_metrics.parquet` | Val-метрики финальной модели |
| `reports/final_reranker_baseline_val_metrics.parquet` | Метрики бейзлайнов |
| `reports/error_analysis_samples.parquet` | 20 конкретных примеров ошибок ранкера |
| `reports/error_analysis_summary.parquet` | Сводка по категориям ошибок |
| `reports/feature_importance.png` | График top-30 важности признаков |
| `reports/v12_feature_importance.csv` | Feature importance (raw) |
| `reports/baseline_comparison.parquet` | Сравнение бейзлайн vs финальная модель |
| `reports/run_config.json` | Конфигурация обучения |
| `reports/versions.json` | Версии библиотек |
| `reports/requirements_snapshot.txt` | Снапшот зависимостей |

### Дологгировать позже (при наличии)
- `robustness_report.parquet` — проверка устойчивости (требует `production_record` в памяти)

## MLflow + MinIO — быстрый старт

### Требования
- Docker Desktop ≥ 4.0 и `docker compose` v2

### Запуск инфраструктуры

```bash
# 1. Скопировать .env
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/Mac

# 2. Запустить MLflow + MinIO
docker compose up -d

# 3. Дождаться ready (обычно 30-60 сек)
docker compose ps               # STATUS должен быть healthy/running
```

| Сервис | URL | Логин |
|--------|-----|-------|
| **MLflow UI** | http://localhost:5000 | — |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| MinIO S3 API | http://localhost:9000 | — |

> Bucket `mlflow-artifacts` создаётся автоматически при старте.

### Остановка

```bash
docker compose down             # остановить (данные сохраняются в Docker volumes)
docker compose down -v          # остановить + удалить volumes
```

### Запуск финального обучения и PRD-логгирования

```bash
# Установить зависимости (если ещё не установлены)
pip install -r requirements.txt

# Открыть ноутбук
jupyter notebook notebooks/5_reranker.ipynb

# Запустить ячейки:
# - 1-32  : обучение реранкера (можно пропустить если артефакты уже есть)
# - 33-34 : reproducibility snapshot (versions.json, run_config.json)
# - 35-36 : MLflow setup (ТРЕБУЕТ запущенного docker compose up -d)
# - 37-38 : логгирование PRD run → артефакты в MinIO
# - 39-40 : baseline comparison (читает из уже посчитанного parquet)
```

### Загрузка PRD-модели (inference)

```bash
jupyter notebook notebooks/6_prd_inference.ipynb
```

### Переменные окружения для ноутбука

```bash
# .env (копия .env.example)
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=recsys_reranker
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

> **Без Docker:** если `MLFLOW_TRACKING_URI` не задан или сервер недоступен,
> ноутбук автоматически использует локальный трекинг (`./mlruns`) — без MinIO.

### Воспроизведение обучения

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить финальный реранкер (все ячейки)
jupyter notebook notebooks/5_reranker.ipynb

# Загрузить PRD-модель и проверить inference
jupyter notebook notebooks/6_prd_inference.ipynb
```

## Запуск сервиса
- должен быть установлен Python 3.10+
- должны быть установлены зависимости из requirements.txt:
    pip install -r requirements.txt
- в корне проекта должна присутствовать директория models с файлами модели, маппингов, популярных товаров и товаров, купленных пользователями ([ссылка](https://disk.yandex.ru/d/dpCwLMmqQjXoJA))
- в корне проекта должен присутствовать файл .env (скопировать содержимое .env.example и изменить JWT_SECRET и ADMIN_PASSWORD)
- запуск сервиса:
    - uvicorn main:app --host 0.0.0.0 --port 8000
- для доступа к административным эндпоинтам и статистике необходимо получить JWT-токен (эндпоинт /login)

### Основные эндпоинты:
- Инференс
    - POST /forward — получить рекомендации для пользователя

- Мониторинг
    - GET /health — liveness-проверка
    - GET /health/ready — readiness-проверка (БД + модель)
    - GET /stats — агрегированные метрики сервиса (admin)
- Администрирование
    - GET /history — история запросов
    - DELETE /history — очистка истории

### Примеры запросов: 
- Инференс рекомендаций:
curl -X POST \
  'http://127.0.0.1:8000/forward' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": 3423970,
    "k": 100
  }'
    - ответ: {
  "user_id": 3423970,
  "recommendations": [12345, 67890, 11223, ...]
}

- Получение JWT-токена (admin):
curl -X POST \
  'http://127.0.0.1:8000/login' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'password=password'
    - ответ: {
    "access_token": "<JWT_TOKEN>",
    "token_type": "bearer"
    }

- Пример запроса с авторизацией:
curl -X GET \
  'http://127.0.0.1:8000/history?limit=100' \
  -H 'Authorization: Bearer <JWT_TOKEN>'
