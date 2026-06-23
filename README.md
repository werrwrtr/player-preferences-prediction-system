
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

│ STEAM API │
│ (GetOwnedGames, GetPlayerSummaries, │
│ GetRecentlyPlayedGames, ResolveVanityURL) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 1. СБОР ДАННЫХ │
│ steam_scraper/ (users, games, details) │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ПРЕДОБРАБОТКА (PySpark) │
│ steam_db.py - очистка, нормализация, merge │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 3. АНАЛИЗ ДАННЫХ │
│ steam.ipynb - TF-IDF, признаки, статистика │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 4. ОБУЧЕНИЕ МОДЕЛЕЙ │
│ ┌─────────────┬─────────────┬─────────────┬────────────┐ │
│ │Content-Based│ PMF │ NeuMF │ LightGCN │ │
│ │ (TF-IDF) │ (Cornac) │ (Cornac) │ (PyTorch) │ │
│ └─────────────┴─────────────┴─────────────┴────────────┘ │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│ 5. ВЕБ-ПРИЛОЖЕНИЕ │
│ app.py (Streamlit) + рекомендация.py │
└─────────────────────────────────────────────────────────────┘

---

## 📁 Структура проекта
steam_recommender_system/
│
├── 1. data_collection/ # Сбор данных через Steam API
│ ├── steam_scraper/ # Модули парсинга
│ │ ├── users_scraper.py
│ │ ├── games_scraper.py
│ │ └── details_scraper.py
│ └── main.py # Главный скрипт сбора
│
├── 2. data_processing/ # Предобработка данных
│ └── steam_db.py # Слияние таблиц в Parquet
│
├── 3. data_analysis/ # Анализ и построение признаков
│ ├── steam.ipynb # Jupyter Notebook с анализом
│ ├── steam_db.py # Обработка данных
│ └── db/ # Финальные датасеты
│ └── final_dataset_*.parquet
│
├── 4. model/ # Обучение моделей
│ ├── content_based_filtering.py # Модель 1: Content-Based
│ ├── train_cornac_models.py # Модели 2-4: PMF, NeuMF, BPR
│ ├── train_lightgcn.py # LightGCN (PyTorch)
│ ├── configs/ # YAML-конфиги для RecBole
│ ├── recbole_data/ # Данные для RecBole
│ └── saved/ # Сохранённые модели
│
├── 5. predictions/ # Предсказания и инференс
│ ├── рекомендация.py # Генерация рекомендаций
│ └── recommender_system/ # Утилиты загрузки
│
├── app.py # Streamlit веб-приложение
├── requirements.txt # Зависимости Python
└── README.md # Этот файл

> ⚠️ **Примечание:** В текущей версии проекта некоторые папки имеют русские названия (`5. предсказания/`, `рекомендация.py`, `требования.txt`). Рекомендуется переименовать их в английские для совместимости с GitHub.

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
2. Создание виртуального окружения
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
