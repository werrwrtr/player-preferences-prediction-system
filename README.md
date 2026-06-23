# Player Preferences Prediction System
Информационная система предсказания предпочтений игроков на основе анализа игровых привычек.
📖 О проекте
Player Preferences Prediction System — это полноценный программный комплекс гибридной рекомендательной системы, разработанный в рамках выпускной квалификационной работы. Система анализирует поведение пользователей Steam и рекомендует игры на основе комбинации коллаборативной фильтрации, контентного анализа и глубокого обучения.
🎯 Цель проекта
Разработать и сравнить несколько подходов к рекомендации игр:
Content-Based Filtering — на основе жанров и тегов игр
Matrix Factorization (PMF) — вероятностная матричная факторизация
Two-Tower Model (NeuMF) — двухбашенная нейронная сеть
LightGCN — графовая нейронная сеть
Гибридная модель — объединение всех подходов
📊 Ключевые особенности
✅ Обработка 2+ миллионов взаимодействий через PySpark
✅ 4 различные модели для сравнения в рамках ВКР
✅ Cold Start через Steam API для новых пользователей
✅ Streamlit веб-интерфейс с визуализацией
✅ MLflow для трекинга экспериментов
✅ Поддержка vanity URL Steam (не только SteamID64)
✨ Возможности
🔍 Для пользователей
🎯 Персональные рекомендации на основе истории игр
🆕 Cold Start — получение рекомендаций для нового пользователя по SteamID
🎮 Информация об играх — обложки, жанры, ссылки на Steam
📊 Аналитический контур — метрики качества рекомендаций
🕒 Игры за 2 недели — отображение недавней активности
🛠️ Для разработчиков
📈 Сравнение моделей — таблица метрик для всех алгоритмов
🧪 MLflow интеграция — трекинг экспериментов
🔄 Модульная архитектура — легко добавлять новые модели
💾 Пайплайн данных — от сбора до инференса
🏗️ Архитектура
┌─────────────────────────────────────────────────────────────┐
│                    STEAM API                                │
│         (GetOwnedGames, GetPlayerSummaries,                 │
│          GetRecentlyPlayedGames, ResolveVanityURL)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              1. DATA COLLECTION                             │
│         steam_scraper/ (users, games, details)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              2. DATA PROCESSING (PySpark)                   │
│         steam_db.py - очистка, нормализация, merge          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              3. DATA ANALYSIS                               │
│         steam.ipynb - TF-IDF, признаки, статистика          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              4. MODEL TRAINING                              │
│   ┌─────────────┬─────────────┬─────────────┬────────────┐ │
│   │Content-Based│     PMF     │    NeuMF    │  LightGCN  │ │
│   │  (TF-IDF)   │   (Cornac)  │   (Cornac)  │  (PyTorch) │ │
│   └─────────────┴─────────────┴─────────────┴────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              5. PREDICTIONS & WEB APP                       │
│         app.py (Streamlit) + recommender.py                 │
└─────────────────────────────────────────────────────────────┘
 Структура проекта
 steam_recommender_system/
├── 1. data_collection/          # Сбор данных через Steam API
│   ├── steam_scraper/           # Модули парсинга
│   │   ├── users_scraper.py
│   │   ├── games_scraper.py
│   │   └── details_scraper.py
│   └── main.py                  # Главный скрипт сбора
│
├── 2. data_processing/          # Предобработка данных
│   └── steam_db.py              # Слияние таблиц в Parquet
│
├── 3. data_analysis/            # Анализ и построение признаков
│   ├── steam.ipynb              # Jupyter notebook с анализом
│   ├── steam_db.py              # Обработка данных
│   └── db/                      # Финальные датасеты
│       └── final_dataset_*.parquet
│
├── 4. model/                    # Обучение моделей
│   ├── content_based_filtering.py    # Модель 1: Content-Based
│   ├── train_cornac_models.py        # Модели 2-4: PMF, NeuMF, BPR
│   ├── train_lightgcn.py             # LightGCN (PyTorch)
│   ├── configs/                      # YAML-конфиги для RecBole
│   ├── recbole_data/                 # Данные для RecBole
│   └── saved/                        # Сохранённые модели
│
├── 5. predictions/              # Предсказания и инференс
│   ├── recommender.py           # Генерация рекомендаций
│   └── recommender_system/      # Утилиты загрузки
│
├── app.py                       # Streamlit веб-приложение
├── requirements.txt             # Зависимости Python
└── README.md                    # Этот файл
🚀 Установка
📋 Требования
Python 3.10+
Java 11+ (для PySpark)
Git
Steam API Key (получить здесь)
💻 Системные требования

Рекомендуется
RAM Минимум  8 GB Рекомендуется 16 GB
CPU Минимум  4 cores Рекомендуется 8 cores
GPU Опционально NVIDIA с CUDA
Диск Минимум 20 GB  Рекомендуется 50 GB SSD
📦 Пошаговая установка
1. Клонирование репозитория bash
git clone https://github.com/yourusername/steam_recommender_system.git
cd steam_recommender_system
2. Создание виртуального окружения
 # Windows
python -m venv .venv
.venv\Scripts\activate
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
3. Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
4. Настройка Steam API
Создайте файл .env в корне проекта:
STEAM_API_KEY=your_steam_api_key_here
5. Проверка установки
python -c "import pyspark, torch, cornac, streamlit; print('✅ Все библиотеки установлены')"
🎯 Запуск
📊 Шаг 1: Сбор данных (опционально)
Если хотите собрать свежие данные из Steam API:
python " сборданных/main.py"
Шаг 2: Предобработка данных
python " обработкаданных/steam_db.py"
Шаг 3: Анализ данных
Откройте 3. data_analysis/steam.ipynb в Jupyter Notebook и выполните все ячейки.
🧠 Шаг 4: Обучение моделей
🌐 Шаг 5: Запуск веб-приложения
Откройте браузер: http://localhost:8501
📚 Технологии
Основные библиотеки
PySpark — обработка больших данных
Cornac — рекомендательные системы
PyTorch — глубокое обучение
Streamlit — веб-интерфейс
scikit-learn — ML утилиты
MLflow — трекинг экспериментов
Дополнительные
Pandas — анализ данных
NumPy — численные вычисления
Requests — HTTP запросы к Steam API
Pillow — обработка изображений
📝 Лицензия
Этот проект распространяется под лицензией MIT. См. LICENSE для подробностей.
🙏 Благодарности
Steam API — за предоставление данных
Cornac Team — за отличную библиотеку для рекомендательных систем
Streamlit Team — за удобный фреймворк для веб-приложений
📧 Контакты
Автор: [Ваше имя]
Email: your.email@example.com
GitHub: @yourusername

# 🎮 Steam Recommender System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-3.5+-E25A1C?logo=apache-spark)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

**Интеллектуальная гибридная рекомендательная система компьютерных игр на основе Steam API**

</div>

---

## 📖 О проекте

**Steam Recommender System** — программный комплекс гибридной рекомендательной системы, разработанный в рамках выпускной квалификационной работы. Система анализирует поведение пользователей Steam и рекомендует игры на основе комбинации коллаборативной фильтрации, контентного анализа и глубокого обучения.

### 🎯 Цель проекта

Разработать и сравнить несколько подходов к рекомендации игр:
- **Content-Based Filtering** — на основе жанров и тегов игр
- **Matrix Factorization (PMF)** — вероятностная матричная факторизация
- **Two-Tower Model (NeuMF)** — двухбашенная нейронная сеть
- **LightGCN** — графовая нейронная сеть
- **Гибридная модель** — объединение всех подходов

### 📊 Ключевые особенности

- ✅ Обработка **2+ миллионов взаимодействий** через PySpark
- ✅ **4 различные модели** для сравнения в рамках ВКР
- ✅ **Cold Start** через Steam API для новых пользователей
- ✅ **Streamlit веб-интерфейс** с визуализацией
- ✅ Поддержка **vanity URL** Steam (не только SteamID64)

---

## 🏗️ Архитектура системы
