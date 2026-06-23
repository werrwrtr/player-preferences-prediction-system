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

Система построена по конвейерному принципу — каждый этап передаёт данные следующему:

| Этап | Компонент | Описание |
|------|-----------|----------|
| **0. Источник** | Steam API | `GetOwnedGames`, `GetPlayerSummaries`, `GetRecentlyPlayedGames`, `ResolveVanityURL` |
| **1. Сбор** | `1. data_collection/` | Парсинг данных о пользователях, играх и деталях |
| **2. Предобработка** | `2. data_processing/` | Очистка, нормализация, слияние таблиц в Parquet |
| **3. Анализ** | `3. data_analysis/` | TF-IDF признаки, статистический анализ |
| **4. Обучение** | `4. model/` | Обучение 4 моделей: Content-Based, PMF, NeuMF, LightGCN |
| **5. Инференс** | `app.py` | Streamlit веб-приложение с рекомендациями |

### Схема потока данных

```
Steam API
    │
    ▼
[1] Сбор данных ──► users.json, games.json, details.json
    │
    ▼
[2] Предобработка (PySpark) ──► final_dataset.parquet
    │
    ▼
[3] Анализ ──► TF-IDF векторы, признаки пользователей
    │
    ▼
[4] Обучение моделей
    ├── Content-Based (TF-IDF + Cosine)
    ├── PMF (Matrix Factorization)
    ├── NeuMF (Two-Tower Neural Network)
    └── LightGCN (Graph Neural Network)
    │
    ▼
[5] Веб-приложение (Streamlit) ──► Персональные рекомендации
```

---

## 📁 Структура проекта

```
steam_recommender_system/
│
├── 1. data_collection/
│   ├── steam_scraper/
│   │   ├── users_scraper.py
│   │   ├── games_scraper.py
│   │   └── details_scraper.py
│   └── main.py
│
├── 2. data_processing/
│   └── steam_db.py
│
├── 3. data_analysis/
│   ├── steam.ipynb
│   ├── steam_db.py
│   └── db/
│       └── final_dataset_*.parquet
│
├── 4. model/
│   ├── content_based_filtering.py
│   ├── train_cornac_models.py
│   ├── train_lightgcn.py
│   ├── configs/
│   ├── recbole_data/
│   └── saved/
│
├── 5. predictions/
│   ├── recommender.py
│   └── recommender_system/
│
├── app.py
├── requirements.txt
└── README.md
```

> ⚠️ **Примечание:** В текущей версии некоторые папки имеют русские названия (`5. предсказания/`, `рекомендация.py`, `требования.txt`). Рекомендуется переименовать их в английские для совместимости с GitHub.

---

## 🚀 Установка

### 📋 Требования

- **Python** 3.10+
- **Java** 11+ (для PySpark)
- **Git**
- **Steam API Key** ([получить здесь](https://steamcommunity.com/dev/apikey))

### 💻 Системные требования

| Компонент | Минимум | Рекомендуется |
|-----------|---------|---------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| GPU | Опционально | NVIDIA с CUDA |
| Диск | 20 GB | 50 GB SSD |

### 📦 Пошаговая установка

#### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/steam_recommender_system.git
cd steam_recommender_system
```

#### 2. Создание виртуального окружения

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Настройка Steam API

Создайте файл `.env` в корне проекта:

```env
STEAM_API_KEY=your_steam_api_key_here
```

#### 5. Проверка установки

```bash
python -c "import pyspark, torch, cornac, streamlit; print('✅ OK')"
```

---

## 🎯 Запуск

### 📊 Шаг 1: Сбор данных (опционально)

```bash
python "1. data_collection/main.py"
```

> ⚠️ Сбор данных занимает несколько часов. Готовый датасет уже включён в репозиторий.

### 🔧 Шаг 2: Предобработка данных

```bash
python "2. data_processing/steam_db.py"
```

### 📈 Шаг 3: Анализ данных

Откройте `3. data_analysis/steam.ipynb` в Jupyter и выполните все ячейки.

### 🧠 Шаг 4: Обучение моделей

```bash
# Модель 1: Content-Based
python "4. model/content_based_filtering.py"

# Модели 2-4: PMF, NeuMF, BPR (через Cornac)
python "4. model/train_cornac_models.py"

# Модель 5: LightGCN (опционально)
python "4. model/train_lightgcn.py"
```

### 🌐 Шаг 5: Запуск веб-приложения

```bash
streamlit run app.py
```

Откройте браузер: **http://localhost:8501**

---

## 🤖 Модели
### 🏗️ Детальное описание

#### 1. Content-Based Filtering
- **Вход:** TF-IDF векторы жанров/тегов (100 dims)
- **Профиль пользователя:** средневзвешенный вектор любимых игр
- **Скор:** косинусное сходство
- **Плюсы:** работает для новых пользователей (Cold Start)
- **Минусы:** не учитывает коллаборативные паттерны

#### 2. PMF (Probabilistic Matrix Factorization)
- **Вход:** матрица user-item взаимодействий
- **Архитектура:** факторизация в user × item эмбеддинги (d=64)
- **Обучение:** ALS с регуляризацией λ=0.1
- **Плюсы:** быстрый, интерпретируемый
- **Минусы:** не учитывает контент

#### 3. NeuMF (Neural Matrix Factorization)
- **Вход:** user_id, item_id
- **Архитектура:** Two-Tower с MLP (128 → 64)
- **Активация:** ReLU + Dropout(0.2)
- **Плюсы:** учитывает нелинейные паттерны
- **Минусы:** требует больше данных

#### 4. LightGCN
- **Вход:** граф взаимодействий user-item
- **Архитектура:** 3 слоя графовой свёртки
- **Агрегация:** конкатенация эмбеддингов со всех слоёв
- **Плюсы:** учитывает структуру графа
- **Минусы:** требует много памяти

---

> 📝 **Примечание:** Низкие абсолютные значения Precision/Recall обусловлены огромным размером каталога (34,161 игр). Модель PMF превосходит случайное угадывание в **18 раз**.

---

## 🌐 Веб-интерфейс

### 🎮 Использование

1. **Выбор пользователя из базы:**
   - Откройте боковую панель
   - Выберите пользователя из выпадающего списка
   - Система покажет персональные рекомендации

2. **Cold Start (новый пользователь):**
   - Переключитесь в режим "Новый пользователь"
   - Введите SteamID64 или vanity URL (например, `sukadedins1241`)
   - Система получит данные через Steam API
   - Покажет игры за последние 2 недели
   - Сгенерирует рекомендации на основе TF-IDF

3. **Аналитика:**
   - Precision@10 — доля релевантных игр в топ-10
   - Hit Rate — процент пользователей с хотя бы одним попаданием
   - NDCG — качество ранжирования

---

## 📚 Технологии

### Основные библиотеки

- **[PySpark](https://spark.apache.org/)** — обработка больших данных
- **[Cornac](https://cornac.readthedocs.io/)** — рекомендательные системы
- **[PyTorch](https://pytorch.org/)** — глубокое обучение
- **[Streamlit](https://streamlit.io/)** — веб-интерфейс
- **[scikit-learn](https://scikit-learn.org/)** — ML утилиты

### Дополнительные

- **[Pandas](https://pandas.pydata.org/)** — анализ данных
- **[NumPy](https://numpy.org/)** — численные вычисления
- **[Requests](https://requests.readthedocs.io/)** — HTTP запросы к Steam API
- **[Pillow](https://python-pillow.org/)** — обработка изображений

---

## 📝 Лицензия

Этот проект распространяется под лицензией MIT.

---

## 📧 Контакты

**Автор:** [Ваше имя]  
**Email:** your.email@example.com  
**GitHub:** [@yourusername](https://github.com/yourusername)

---

<div align="center">

**⭐ Если проект был полезен, поставьте звезду на GitHub!**

</div>
