#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNIFIED STEAM DATA COLLECTOR AND ANALYZER
=========================================
Полный пайплайн сбора и анализа данных Steam:
1. Сбор друзей и уникальных ID
2. Сбор игр пользователей
3. Сбор деталей пользователей
4. Сбор достижений
5. Сбор деталей игр (скрапинг)
6. Анализ стилей геймплея (гибридный)
7. Анализ трендов и сегментация
"""

import os
import sys
import glob
import json
import time
import re
import random
import traceback
import argparse
from datetime import datetime
from collections import Counter, defaultdict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import requests

# ==========================================
# НАСТРОЙКА ПУТЕЙ
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) in ['1. data_collection', 'data_collection', 'src']:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    PROJECT_ROOT = SCRIPT_DIR

os.chdir(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
class Config:
    # Цели сбора
    TARGET_OPEN_PROFILES = 19384
    CHECKPOINT_INTERVAL = 2000
    BATCH_SIZE = 100
    MAX_WORKERS = 6
    
    # Минимальное время игры (минуты)
    MIN_PLAYTIME_MINUTES = 60
    
    # Сегментация (часы за 2 недели)
    SEGMENT_THRESHOLDS = {
        'hardcore': 100,
        'regular': 20,
        'casual': 5,
        'light': 0,
        'inactive': None,
    }
    
    # Пути
    DB_PATH = os.path.join(PROJECT_ROOT, 'db')
    
    @classmethod
    def get_db_path(cls, subdir=''):
        path = os.path.join(cls.DB_PATH, subdir)
        os.makedirs(path, exist_ok=True)
        return path

# ==========================================
# МАППИНГ ЖАНРОВ И ТЕГОВ (для анализа стилей)
# ==========================================

GENRE_TO_STYLE = {
    'Action': ['competitive'],
    'Sports': ['competitive'],
    'Racing': ['competitive'],
    'Adventure': ['explorer'],
    'Casual': ['explorer'],
    'Early Access': ['explorer'],
    'Indie': ['explorer'],
    'Education': ['explorer'],
    'Simulation': ['completionist', 'grinder'],
    'Strategy': ['completionist'],
    'RPG': ['completionist', 'explorer'],
    'Free To Play': ['social'],
    'Massively Multiplayer': ['social'],
}

TAG_TO_STYLE = {
    'Multiplayer': ['social'],
    'Co-op': ['social'],
    'Cooperative': ['social'],
    'Online Co-Op': ['social'],
    'Local Co-Op': ['social'],
    'Local Multiplayer': ['social'],
    'Asynchronous Multiplayer': ['social'],
    '4 Player Local': ['social'],
    'Massively Multiplayer': ['social'],
    'MMO': ['social'],
    'MMORPG': ['social'],
    'Team-Based': ['social'],
    'Guild': ['social'],
    'Social Deduction': ['social'],
    'Online PvP': ['social'],
    'PvP': ['social'],
    'PVP': ['social'],
    'Fighting': ['social', 'competitive'],
    'Party': ['social'],
    'Comedy': ['social'],
    'Free to Play': ['social'],
    'Funny': ['social'],
    'Roguelike': ['grinder'],
    'Roguelite': ['grinder'],
    'Action Roguelike': ['grinder'],
    'Survival': ['grinder'],
    'Crafting': ['grinder'],
    'Sandbox': ['grinder', 'completionist'],
    'Farming': ['grinder'],
    'Management': ['grinder', 'completionist'],
    'Base Building': ['grinder'],
    'Resource Management': ['grinder'],
    'Hack and Slash': ['grinder'],
    'Loot': ['grinder'],
    'Grind': ['grinder'],
    'Procedural Generation': ['grinder'],
    'Replayability': ['grinder'],
    'Character Customization': ['grinder'],
    'Building': ['grinder'],
    'Singleplayer': ['explorer'],
    'Adventure': ['explorer'],
    'Indie': ['explorer'],
    'Casual': ['explorer'],
    'Horror': ['explorer'],
    'Relaxing': ['explorer'],
    'Action-Adventure': ['explorer'],
    'Early Access': ['explorer'],
    'Female Protagonist': ['explorer'],
    'Psychological Horror': ['explorer'],
    'Mystery': ['explorer'],
    'Detective': ['explorer'],
    'Metroidvania': ['explorer'],
    'Hidden Object': ['explorer'],
    '2D': ['explorer'],
    'Open World': ['explorer', 'completionist'],
    'Exploration': ['explorer', 'completionist'],
    'Story Rich': ['explorer', 'completionist'],
    'Atmospheric': ['explorer', 'completionist'],
    'Fantasy': ['explorer'],
    'Sci-fi': ['explorer'],
    'Anime': ['explorer'],
    'Family Friendly': ['explorer', 'completionist'],
    'Colorful': ['explorer'],
    'Cute': ['explorer'],
    'Collectathon': ['completionist'],
    'Achievements': ['completionist'],
    'Completionist': ['completionist'],
    '100%': ['completionist'],
    'Challenges': ['completionist'],
    'Trophy': ['completionist'],
    'Simulation': ['completionist', 'grinder'],
    'Strategy': ['completionist'],
    'RPG': ['completionist', 'explorer'],
    'Difficult': ['completionist', 'grinder'],
    'Puzzle': ['grinder', 'explorer'],
    'Competitive': ['competitive'],
    'FPS': ['competitive'],
    'Shooter': ['competitive'],
    'MOBA': ['competitive'],
    'Battle Royale': ['competitive'],
    'Arcade': ['competitive'],
    'eSports': ['competitive'],
    'Action': ['competitive'],
    'First-Person': ['competitive'],
    'Realistic': ['competitive'],
    'Gore': ['competitive'],
    'Third Person': ['competitive'],
}

TAG_WEIGHTS = {
    'Singleplayer': 0.3, 'Indie': 0.3, 'Casual': 0.3, 'Adventure': 0.5,
    '2D': 0.5, 'Colorful': 0.5, 'Cute': 0.5, 'Early Access': 0.5,
    'Relaxing': 0.5, 'Action': 0.7, 'RPG': 0.8, 'Fantasy': 0.7,
    'Sci-fi': 0.7, 'Horror': 0.8, 'Atmospheric': 0.7, 'Story Rich': 0.8,
    'Open World': 0.8, 'Exploration': 0.8, 'Family Friendly': 0.6,
    'Female Protagonist': 0.6, 'Psychological Horror': 0.8, 'Metroidvania': 0.9,
    'Action-Adventure': 0.7, 'Anime': 0.7,
    'Survival': 2.5, 'Roguelike': 2.5, 'Roguelite': 2.5, 'Action Roguelike': 2.5,
    'Crafting': 2.5, 'Farming': 2.5, 'Hack and Slash': 2.0, 'Loot': 2.5,
    'Grind': 3.0, 'Procedural Generation': 2.0, 'Replayability': 2.0,
    'Base Building': 2.0, 'Resource Management': 2.0, 'Building': 1.8,
    'Character Customization': 1.5, 'Sandbox': 1.5, 'Management': 1.5,
    'Simulation': 1.5, 'Strategy': 1.3, 'Difficult': 1.5, 'Puzzle': 1.2,
    'Collectathon': 2.0, 'Achievements': 2.0, 'Completionist': 2.5,
    '100%': 2.5, 'Challenges': 1.8, 'Trophy': 2.0,
    'Competitive': 1.5, 'FPS': 1.3, 'Shooter': 1.2, 'MOBA': 1.5,
    'Battle Royale': 1.5, 'Arcade': 1.0, 'eSports': 2.0, 'First-Person': 0.8,
    'Realistic': 1.0, 'Gore': 1.0, 'Third Person': 0.7,
    'Multiplayer': 1.0, 'Co-op': 1.2, 'Cooperative': 1.2, 'Online Co-Op': 1.2,
    'Local Co-Op': 1.3, 'Local Multiplayer': 1.3, 'Massively Multiplayer': 1.3,
    'MMO': 1.5, 'MMORPG': 1.5, 'Team-Based': 1.2, 'PvP': 1.3,
    'PVP': 1.3, 'Online PvP': 1.3, 'Fighting': 1.2, 'Party': 1.3,
    'Free to Play': 0.8, 'Funny': 0.7, 'Comedy': 0.7,
}

GRINDER_TAGS = {
    'Roguelike', 'Roguelite', 'Action Roguelike', 'Survival', 'Crafting',
    'Sandbox', 'Farming', 'Management', 'Base Building', 'Resource Management',
    'Hack and Slash', 'Loot', 'Grind', 'Procedural Generation', 'Replayability',
    'Character Customization', 'Building'
}

STYLES = ['completionist', 'competitive', 'explorer', 'social', 'grinder']

# ==========================================
# УТИЛИТЫ
# ==========================================

def load_latest_json(directory, prefix):
    """Загружает последний валидный JSON-файл с заданным префиксом."""
    full_path = os.path.join(PROJECT_ROOT, directory)
    if not os.path.exists(full_path):
        return None
    files = glob.glob(os.path.join(full_path, f'{prefix}_*.json'))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded: {os.path.basename(filepath)}")
                return data
        except json.JSONDecodeError:
            continue
    return None

def save_json(data, directory, prefix, is_checkpoint=False):
    """Сохраняет данные с временной меткой или как контрольную точку."""
    full_path = os.path.join(PROJECT_ROOT, directory)
    os.makedirs(full_path, exist_ok=True)
    
    if is_checkpoint:
        filepath = os.path.join(full_path, f'{prefix}_checkpoint.json')
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filepath = os.path.join(full_path, f'{prefix}_{timestamp}.json')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if not is_checkpoint:
        print(f"Saved: {filepath}")
    return filepath

def load_all_games_details():
    """Загружает ВСЕ файлы games_detail из папки db/games_detail и объединяет."""
    full_path = os.path.join(PROJECT_ROOT, 'db/games_detail')
    if not os.path.exists(full_path):
        return {}
    
    files = glob.glob(os.path.join(full_path, '*.json'))
    combined = {}
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            appid = item.get('appid') or item.get('gameid') or item.get('id')
                            if appid:
                                combined[str(appid)] = item
                elif isinstance(data, dict):
                    for k, v in data.items():
                        combined[str(k)] = v
        except Exception:
            continue
    
    return combined

def load_all_games_list():
    """Загружает последний файл games из db/games."""
    full_path = os.path.join(PROJECT_ROOT, 'db/games')
    if not os.path.exists(full_path):
        return {}
    files = glob.glob(os.path.join(full_path, 'games_*.json'))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            continue
    return None

def is_valid_profile(games_list):
    """Проверяет, является ли профиль открытым и содержащим игры."""
    if not isinstance(games_list, list) or len(games_list) == 0:
        return False
    if len(games_list) == 1 and isinstance(games_list[0], dict) and games_list[0].get('private'):
        return False
    return True

def get_ids_list(data):
    """Безопасное получение списка ID."""
    if isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, list):
        return data
    return []

def get_segment(playtime_hours):
    """Определяет сегмент пользователя по времени игры за 2 недели."""
    thresholds = Config.SEGMENT_THRESHOLDS
    if playtime_hours >= thresholds['hardcore']:
        return 'hardcore'
    elif playtime_hours >= thresholds['regular']:
        return 'regular'
    elif playtime_hours >= thresholds['casual']:
        return 'casual'
    elif playtime_hours > 0:
        return 'light'
    else:
        return 'inactive'

def build_games_name_map(games_list):
    """Строит словарь appid -> name из списка игр всех пользователей."""
    name_map = {}
    if not games_list:
        return name_map
    
    for steamid, user_games in games_list.items():
        if not isinstance(user_games, list):
            continue
        for game in user_games:
            appid = game.get('appid')
            name = game.get('name')
            if appid and name and str(appid) not in name_map:
                name_map[str(appid)] = name
    
    return name_map

def get_game_name(games_name_map, games_detail, appid):
    """Получает название игры по appid из нескольких источников."""
    appid_str = str(appid)
    
    if appid_str in games_name_map:
        return games_name_map[appid_str]
    
    detail = games_detail.get(appid_str)
    if detail:
        for field in ['name', 'title', 'game_name']:
            if field in detail and detail[field]:
                return detail[field]
        if 'common' in detail and isinstance(detail['common'], dict):
            name = detail['common'].get('name')
            if name:
                return name
        if 'data' in detail and isinstance(detail['data'], dict):
            name = detail['data'].get('name')
            if name:
                return name
    
    return f"AppID {appid}"

def add_style_scores(style_scores, total_weight, styles, weighted_game_weight):
    """Добавляет очки во все стили из списка с учётом веса."""
    if isinstance(styles, str):
        styles = [styles]
    for style in styles:
        if style in style_scores:
            style_scores[style] += weighted_game_weight
            total_weight += weighted_game_weight
    return total_weight

def normalize_scores(scores):
    """Нормализует словарь очков в проценты (сумма = 1.0)."""
    total = sum(scores.values())
    if total > 0:
        return {s: v / total for s, v in scores.items()}
    return {s: 0.0 for s in scores}

def get_dominant(scores):
    """Возвращает доминирующий стиль."""
    dominant = max(scores, key=scores.get)
    if scores[dominant] == 0:
        return 'unknown'
    return dominant

def process_game(detail, game_weight):
    """Обрабатывает одну игру и возвращает очки по стилям."""
    style_scores = {s: 0.0 for s in STYLES}
    total_weight = 0.0
    is_grinder_game = False

    # Жанры
    genres = detail.get('genres') or detail.get('genre') or []
    if isinstance(genres, list):
        for g in genres:
            genre_name = g if isinstance(g, str) else g.get('description', '')
            styles = GENRE_TO_STYLE.get(genre_name)
            if styles:
                total_weight = add_style_scores(style_scores, total_weight, styles, game_weight)

    # Теги
    tags = detail.get('tags', [])
    if isinstance(tags, list):
        for t in tags:
            tag_name = t if isinstance(t, str) else t.get('name', '')
            styles = TAG_TO_STYLE.get(tag_name)
            if styles:
                tag_weight_mult = TAG_WEIGHTS.get(tag_name, 1.0)
                weighted_game_weight = game_weight * tag_weight_mult
                total_weight = add_style_scores(style_scores, total_weight, styles, weighted_game_weight)
                if tag_name in GRINDER_TAGS:
                    is_grinder_game = True

    return style_scores, total_weight, is_grinder_game

# ==========================================
# МОДУЛЬ: СБОР ДРУЗЕЙ (из main.py)
# ==========================================

def fetch_friend_data(api_key, initial_steamid, max_unique_ids, max_workers=10):
    """Собирает данные о друзьях в социальном графе."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def get_friends_data(api_key, steamid):
        url = 'https://api.steampowered.com/ISteamUser/GetFriendList/v0001/'
        try:
            response = requests.get(
                url,
                params={'key': api_key, 'steamid': steamid, 'relationship': 'all'},
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return steamid, data.get('friendslist', {}).get('friends', [])
        except Exception as e:
            print(f'Error fetching data for {steamid}: {e}')
            return steamid, []

    unique_ids = set()
    friends_data = []
    steamid_queue = [initial_steamid]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=max_unique_ids, desc="Collecting users") as pbar:
            while steamid_queue and len(unique_ids) < max_unique_ids:
                batch = []
                while steamid_queue and len(batch) < max_workers:
                    steamid = steamid_queue.pop(0)
                    if steamid not in unique_ids:
                        batch.append(steamid)

                futures = [executor.submit(get_friends_data, api_key, sid) for sid in batch]

                for future in as_completed(futures):
                    steamid, friends = future.result()
                    if steamid in unique_ids:
                        continue

                    unique_ids.add(steamid)
                    pbar.update(1)

                    friends_data.append({
                        'parent_steamid': steamid,
                        'friendslist': {'friends': friends}
                    })

                    for friend in friends:
                        if friend['steamid'] not in unique_ids and len(unique_ids) + len(steamid_queue) < max_unique_ids:
                            steamid_queue.append(friend['steamid'])

    return {'friends_list': friends_data}, list(unique_ids)

# ==========================================
# МОДУЛЬ: СБОР ИГР (из main.py)
# ==========================================

def get_games_data(**kwargs):
    """Получает игры пользователя через Steam API."""
    url = 'https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/'
    response = requests.get(url, params=kwargs)
    return response

def fetch_user_games(api_key, steamid, max_retries=3):
    """Запрашивает игры пользователя с повторными попытками."""
    for attempt in range(max_retries):
        try:
            response = get_games_data(
                key=api_key,
                steamid=steamid,
                include_appinfo=True,
                include_played_free_games=True,
                include_playtime_2weeks=True
            )
            if response.status_code == 200:
                data = response.json()
                return steamid, data.get('response', {}).get('games', [])
            elif response.status_code == 403:
                return steamid, [{'private': True}]
            elif response.status_code == 429:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                return steamid, []
        except Exception:
            return steamid, []
    return steamid, []

# ==========================================
# МОДУЛЬ: ДЕТАЛИ ПОЛЬЗОВАТЕЛЕЙ (из main.py)
# ==========================================

def get_users_detail(**kwargs):
    """Получает детали пользователей через Steam API."""
    url = 'https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/'
    response = requests.get(url, params=kwargs)
    return response

def fetch_user_detail(api_key, steamid, max_retries=3):
    """Запрашивает детали пользователя с повторными попытками."""
    for attempt in range(max_retries):
        try:
            response = get_users_detail(key=api_key, steamids=steamid)
            if response.status_code == 200:
                data = response.json()
                players = data.get('response', {}).get('players', [])
                if players:
                    return steamid, players[0]
            elif response.status_code == 429:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                break
        except Exception:
            break
    return steamid, None

# ==========================================
# МОДУЛЬ: ДЕТАЛИ ИГР (скрапинг)
# ==========================================

def scrape_game_detail(game_id):
    """Скрапит детали игры со страницы Steam."""
    url = f'https://store.steampowered.com/app/{game_id}'
    
    cookies = {
        'wants_mature_content': '1',
        'birthtime': '567993601',
        'lastagecheckage': '1-January-1990',
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }
    
    try:
        response = requests.get(url, cookies=cookies, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        result = {'gameid': game_id}
        
        # Дата релиза
        try:
            release_div = soup.find('div', class_='release_date')
            if release_div:
                date_div = release_div.find('div', class_='date')
                if date_div:
                    date_text = date_div.get_text(strip=True)
                    if date_text and 'Coming Soon' not in date_text:
                        try:
                            result['release_date'] = datetime.strptime(date_text, '%d %b, %Y').strftime('%Y-%m-%d')
                        except ValueError:
                            result['release_date'] = date_text
                    else:
                        result['release_date'] = 'TBD'
        except Exception:
            result['release_date'] = None
        
        # Разработчик и издатель
        dev_rows = soup.find_all('div', class_='dev_row')
        dev_data = {}
        for row in dev_rows:
            subtitle_div = row.find('div', class_='subtitle column')
            if subtitle_div:
                subtitle = subtitle_div.get_text(strip=True).rstrip(':')
                summary_div = row.find('div', class_='summary column')
                if summary_div:
                    value = summary_div.get_text(strip=True)
                    if value:
                        dev_data[subtitle] = value
        result['developer'] = dev_data.get('Developer')
        result['publisher'] = dev_data.get('Publisher')
        
        # Жанры
        try:
            genres_block = soup.find('div', class_='details_block')
            if genres_block:
                genres_span = genres_block.find('span', attrs={'data-panel': '{"flow-children":"row"}'})
                if genres_span:
                    result['genres'] = [a.get_text(strip=True) for a in genres_span.find_all('a')]
                else:
                    result['genres'] = []
        except Exception:
            result['genres'] = []
        
        # Теги
        try:
            rightcol = soup.find('div', class_='rightcol')
            if rightcol:
                tags_div = rightcol.find('div', class_='glance_tags popular_tags')
                if tags_div:
                    result['tags'] = [a.get_text(strip=True) for a in tags_div.find_all('a', class_='app_tag')]
                else:
                    result['tags'] = []
        except Exception:
            result['tags'] = []
        
        # Рейтинг
        try:
            if rightcol:
                rating_div = rightcol.find('div', class_='user_reviews')
                if rating_div:
                    summary_div = rating_div.find('div', class_='summary column')
                    if summary_div:
                        rating_span = summary_div.find('span')
                        if rating_span:
                            rating_text = rating_span.get_text(strip=True)
                            if rating_text:
                                result['rating'] = rating_text
        except Exception:
            result['rating'] = None
        
        # Название игры
        game_title = soup.find('div', class_='apphub_AppName')
        if game_title:
            result['name'] = game_title.get_text(strip=True)
        
        return result
        
    except Exception as e:
        print(f"Error scraping game {game_id}: {e}")
        return None

def scrape_games_batch(game_ids, max_workers=10):
    """Скрапит пачку игр параллельно."""
    results = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scrape_game_detail, gid): gid for gid in game_ids}
        for future in tqdm(as_completed(futures), total=len(game_ids), desc='Scraping games'):
            game_id = futures[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
                else:
                    failed.append(game_id)
            except Exception:
                failed.append(game_id)
    
    return results, failed

def convert_games_list_to_dict(games_list):
    """Преобразует список словарей в словарь {appid: game_data}."""
    result = {}
    for game in games_list:
        if not isinstance(game, dict):
            continue
        appid = game.get('gameid') or game.get('appid') or game.get('id')
        if appid:
            result[str(appid)] = game
    return result

# ==========================================
# МОДУЛЬ: СБОР ДОСТИЖЕНИЙ
# ==========================================

def get_achievements(api_key, steamid, appid, session):
    """Запрашивает достижения для одной игры пользователя."""
    url = 'https://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v1/'
    params = {'key': api_key, 'steamid': str(steamid), 'appid': str(appid)}
    try:
        response = session.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            if data.get('playerstats', {}).get('success', False):
                return data['playerstats'].get('achievements', [])
    except Exception:
        pass
    return None

# ==========================================
# МОДУЛЬ: АНАЛИЗ СТИЛЕЙ
# ==========================================

class StyleAnalyzer:
    """Анализирует игровой стиль на основе достижений и жанров."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        # Паттерны для анализа достижений
        self.style_patterns = {
            'completionist': re.compile(r'\b(?:complete|collect|all|finish|100\s*%|mastery)\b', re.IGNORECASE),
            'competitive': re.compile(r'\b(?:win|beat|expert|rank|first|undefeated|champion)\b', re.IGNORECASE),
            'explorer': re.compile(r'\b(?:discover|explore|secret|hidden|find|visit|uncover)\b', re.IGNORECASE),
            'social': re.compile(r'\b(?:friend|multiplayer|coop|team|online|party|guild)\b', re.IGNORECASE),
            'grinder': re.compile(r'\b(?:kill|play|hours|reach|earn|level|farm|grind)\b', re.IGNORECASE)
        }
    
    def get_achievements(self, steamid, appid):
        """Запрашивает достижения через API."""
        return get_achievements(self.api_key, steamid, appid, self.session)
    
    def analyze_achievements(self, achievements, playtime_2weeks=0, playtime_forever=0):
        """Анализирует достижения и возвращает вектор стилей."""
        scores = {s: 0.0 for s in STYLES}
        earned = [a for a in achievements if a.get('achieved', 0) == 1]
        if not earned:
            return scores, 0.0
        
        hours_2w = playtime_2weeks / 60.0
        hours_forever = playtime_forever / 60.0
        weight = 1.0 + np.log1p(hours_2w) * 1.5 + np.log1p(hours_forever) * 0.1
        
        for ach in earned:
            text = f"{ach.get('name', '')} {ach.get('description', '')}"
            for style, pattern in self.style_patterns.items():
                if pattern.search(text):
                    scores[style] += weight
        
        return scores, weight * len(earned)
    
    def process_user(self, steamid, games, games_detail, checked_cache, achievement_cache):
        """Обрабатывает одного пользователя и возвращает профиль."""
        if not games or not isinstance(games, list):
            return None
        
        steamid_str = str(steamid)
        
        # Выбор игр
        games_with_2w = [g for g in games if g.get('playtime_2weeks', 0) > 0]
        games_with_2w_sorted = sorted(games_with_2w, key=lambda x: x.get('playtime_2weeks', 0), reverse=True)
        other_games = [g for g in games if g.get('playtime_2weeks', 0) == 0]
        other_games_sorted = sorted(other_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
        
        selected = games_with_2w_sorted[:5] + other_games_sorted[:10]
        if len(selected) < 3:
            all_sorted = sorted(games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
            for g in all_sorted:
                if g not in selected and len(selected) < 15:
                    selected.append(g)
        
        # Анализ
        style_scores = {s: 0.0 for s in STYLES}
        total_weight = 0.0
        games_with_achievements = 0
        games_analyzed = 0
        total_playtime_2w = 0
        
        for game in selected:
            appid = game.get('appid')
            if not appid:
                continue
            appid_str = str(appid)
            playtime_2w = game.get('playtime_2weeks', 0)
            playtime_forever = game.get('playtime_forever', 0)
            total_playtime_2w += playtime_2w
            
            # Проверка кэша достижений
            if steamid_str in checked_cache and appid_str in achievement_cache.get(steamid_str, {}):
                ach_data = achievement_cache[steamid_str][appid_str]
                if ach_data and isinstance(ach_data, dict) and not ach_data.get('no_data'):
                    scores, _ = self.analyze_achievements(
                        ach_data.get('achievements', []), playtime_2w, playtime_forever
                    )
                    game_weight = 1.0 + np.log1p(playtime_2w / 60) * 1.5 + np.log1p(playtime_forever / 60) * 0.1
                    for s in STYLES:
                        style_scores[s] += scores[s] * game_weight
                    total_weight += game_weight
                    games_with_achievements += 1
                    games_analyzed += 1
                    continue
                else:
                    continue
            
            # Запрос достижений
            achievements = self.get_achievements(steamid, appid)
            if achievements is not None:
                if steamid_str not in achievement_cache:
                    achievement_cache[steamid_str] = {}
                achievement_cache[steamid_str][appid_str] = {'achievements': achievements, 'no_data': False}
                checked_cache[steamid_str] = True
                
                scores, _ = self.analyze_achievements(achievements, playtime_2w, playtime_forever)
                game_weight = 1.0 + np.log1p(playtime_2w / 60) * 1.5 + np.log1p(playtime_forever / 60) * 0.1
                for s in STYLES:
                    style_scores[s] += scores[s] * game_weight
                total_weight += game_weight
                games_with_achievements += 1
                games_analyzed += 1
            else:
                # Fallback на жанры
                detail = games_detail.get(appid_str)
                if detail:
                    genres = detail.get('genres', [])
                    if genres:
                        genre_scores = {s: 0.0 for s in STYLES}
                        for genre in genres:
                            styles = GENRE_TO_STYLE.get(genre, [])
                            for style in styles:
                                if style in genre_scores:
                                    genre_scores[style] += 1.0
                        total_genre = sum(genre_scores.values())
                        if total_genre > 0:
                            for s in genre_scores:
                                genre_scores[s] = genre_scores[s] / total_genre
                            game_weight = np.log1p(playtime_forever / 60) if playtime_forever > 0 else 0.5
                            if game_weight > 0:
                                for s in STYLES:
                                    style_scores[s] += genre_scores[s] * game_weight
                                total_weight += game_weight
                                games_analyzed += 1
        
        # Формирование профиля
        profile = {
            'steamid': steamid_str,
            'total_games': len(games),
            'games_analyzed': games_analyzed,
            'games_with_achievements': games_with_achievements,
            'playtime_2weeks_minutes': total_playtime_2w,
            'playtime_2weeks_hours': round(total_playtime_2w / 60.0, 2),
            **{s: 0.0 for s in STYLES},
            'dominant_style': 'unknown'
        }
        
        if total_weight > 0:
            for s in STYLES:
                profile[s] = round(style_scores[s] / total_weight, 3)
            style_vals = {s: profile[s] for s in STYLES}
            dominant = max(style_vals, key=style_vals.get)
            if style_vals[dominant] > 0:
                profile['dominant_style'] = dominant
        
        return profile

# ==========================================
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Unified Steam Data Collector and Analyzer')
    parser.add_argument('--force-step', type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help='Принудительно начать с указанного шага')
    parser.add_argument('--skip-achievements', action='store_true',
                        help='Пропустить сбор достижений')
    parser.add_argument('--skip-game-details', action='store_true',
                        help='Пропустить скрапинг деталей игр')
    parser.add_argument('--skip-style-analysis', action='store_true',
                        help='Пропустить анализ стилей')
    parser.add_argument('--skip-trends', action='store_true',
                        help='Пропустить анализ трендов')
    args = parser.parse_args()
    
    # Загрузка .env
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
    else:
        print(f"Warning: .env not found at {env_path}")
    
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("ERROR: API_KEY not found in .env!")
        print("Continue without API_KEY? Some features will be disabled.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    total_start_time = time.time()
    
    print('\n' + '=' * 70)
    print('UNIFIED STEAM DATA COLLECTOR AND ANALYZER')
    print('=' * 70)
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Python version: {sys.version}')
    print('=' * 70)
    
    # ==========================================
    # АНАЛИЗ ТЕКУЩЕГО СОСТОЯНИЯ
    # ==========================================
    print('\nAnalyzing current database state...')
    
    friends_data = load_latest_json('db/users', 'users')
    unique_ids_data = load_latest_json('db/unique_ids', 'unique_ids')
    games_data = load_latest_json('db/games', 'games')
    user_details = load_latest_json('db/users_detail', 'users_detail')
    game_details = load_all_games_details()
    achievements_cache = load_latest_json('db/achievements', 'achievements') or {}
    
    print(f"\nCurrent state:")
    print(f"  - friends_data: {'exists' if friends_data else 'missing'}")
    print(f"  - unique_ids_data: {'exists' if unique_ids_data else 'missing'}")
    print(f"  - games_data: {'exists' if games_data else 'missing'}")
    print(f"  - user_details: {'exists' if user_details else 'missing'}")
    print(f"  - game_details: {len(game_details)} records")
    print(f"  - achievements_cache: {len(achievements_cache)} users")
    
    # Определяем стартовый шаг
    start_step = 1
    if args.force_step:
        start_step = args.force_step
        print(f'\nForced start from step {start_step}')
    else:
        if friends_data and unique_ids_data:
            start_step = max(start_step, 2)
        if games_data and isinstance(games_data, dict):
            valid_count = sum(1 for g in games_data.values() if is_valid_profile(g))
            if valid_count >= Config.TARGET_OPEN_PROFILES:
                start_step = max(start_step, 3)
        if user_details and isinstance(user_details, dict):
            start_step = max(start_step, 4)
        if achievements_cache:
            start_step = max(start_step, 5)
        if game_details:
            start_step = max(start_step, 6)
        
        print(f'\nAuto-detected start step: {start_step}')
    
    # ==========================================
    # ШАГ 1: СБОР ДРУЗЕЙ
    # ==========================================
    if start_step <= 1:
        print('\n' + '=' * 70)
        print('STEP 1: Collecting friends and unique SteamIDs')
        print('=' * 70)
        step_start = time.time()
        
        if not api_key:
            print("ERROR: API_KEY required for this step")
            return
        
        if friends_data is None or unique_ids_data is None:
            print('Starting full collection...')
            friends_data, unique_steamids = fetch_friend_data(
                api_key=api_key,
                initial_steamid='76561198147702705',
                max_unique_ids=260000
            )
            save_json(friends_data, 'db/users', 'users')
            save_json(unique_steamids, 'db/unique_ids', 'unique_ids')
        else:
            unique_steamids = unique_ids_data
            print(f'Data already exists ({len(get_ids_list(unique_steamids))} profiles)')
        
        print(f'Step 1 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    all_target_steamids = get_ids_list(unique_ids_data) if unique_ids_data else []
    all_target_steamids_set = set(all_target_steamids)
    
    # ==========================================
    # ШАГ 2: СБОР ИГР
    # ==========================================
    if start_step <= 2:
        print('\n' + '=' * 70)
        print('STEP 2: Collecting user games')
        print('=' * 70)
        step_start = time.time()
        
        if not api_key:
            print("ERROR: API_KEY required for this step")
            return
        
        if games_data is None or not isinstance(games_data, dict):
            games_data = {}
        
        current_valid = sum(1 for g in games_data.values() if is_valid_profile(g))
        remaining = max(0, Config.TARGET_OPEN_PROFILES - current_valid)
        
        print(f"Current: {current_valid} open profiles")
        print(f"Target: {Config.TARGET_OPEN_PROFILES}. Remaining: {remaining}")
        
        existing_ids = set(games_data.keys())
        missing_ids = list(all_target_steamids_set - existing_ids)
        
        if remaining == 0:
            print('Target reached! Skipping step 2.')
        elif not missing_ids:
            print('All users already checked.')
        else:
            print(f"Checking {len(missing_ids)} new users...")
            
            processed = 0
            new_valid = 0
            private_count = 0
            goal_reached = False
            
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_user_games, api_key, sid): sid for sid in missing_ids}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc='Checking profiles'):
                    steamid, user_games = future.result()
                    games_data[steamid] = user_games
                    processed += 1
                    
                    if is_valid_profile(user_games):
                        new_valid += 1
                    elif user_games and user_games[0].get('private'):
                        private_count += 1
                    
                    total_valid_now = current_valid + new_valid
                    
                    if total_valid_now >= Config.TARGET_OPEN_PROFILES and not goal_reached:
                        print(f"\nTarget reached! Open: {total_valid_now}, Private: {private_count}")
                        goal_reached = True
                        save_json(games_data, 'db/games', 'games')
                        break
                    
                    if processed % Config.CHECKPOINT_INTERVAL == 0 and not goal_reached:
                        save_json(games_data, 'db/games', 'games', is_checkpoint=True)
            
            if not goal_reached:
                save_json(games_data, 'db/games', 'games')
            
            print(f"Step 2 result: +{new_valid} open, {private_count} private")
        
        print(f'Step 2 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    # Извлекаем уникальные appid
    unique_appids = set()
    if games_data:
        for games_list in games_data.values():
            if is_valid_profile(games_list):
                for game in games_list:
                    if isinstance(game, dict) and 'appid' in game:
                        unique_appids.add(str(game['appid']))
    
    print(f"Unique appids in profiles: {len(unique_appids)}")
    
    # ==========================================
    # ШАГ 3: ДЕТАЛИ ПОЛЬЗОВАТЕЛЕЙ
    # ==========================================
    if start_step <= 3:
        print('\n' + '=' * 70)
        print('STEP 3: Collecting user details')
        print('=' * 70)
        step_start = time.time()
        
        if not api_key:
            print("ERROR: API_KEY required for this step")
            return
        
        if user_details is None or not isinstance(user_details, dict):
            user_details = {}
        
        existing_ids = set(user_details.keys())
        valid_steamids = {sid for sid, g in games_data.items() if is_valid_profile(g)} if games_data else set()
        missing_ids = list(valid_steamids - existing_ids)
        
        if not missing_ids:
            print(f'Details already collected for all {len(valid_steamids)} users.')
        else:
            print(f"Collecting details for {len(missing_ids)} new users...")
            
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_user_detail, api_key, sid): sid for sid in missing_ids}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching user details'):
                    steamid, player_data = future.result()
                    if player_data:
                        user_details[steamid] = player_data
            
            save_json(user_details, 'db/users_detail', 'users_detail')
        
        print(f'Step 3 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    # ==========================================
    # ШАГ 4: СБОР ДОСТИЖЕНИЙ
    # ==========================================
    if start_step <= 4 and not args.skip_achievements:
        print('\n' + '=' * 70)
        print('STEP 4: Collecting achievements')
        print('=' * 70)
        step_start = time.time()
        
        if not api_key:
            print("ERROR: API_KEY required for this step")
            return
        
        if achievements_cache is None:
            achievements_cache = {}
        checked_users = load_latest_json('db', 'checked_users') or {}
        
        valid_users = {sid: games for sid, games in games_data.items() if is_valid_profile(games)} if games_data else {}
        print(f"Open profiles: {len(valid_users)}")
        print(f"Cache contains achievements for {len(achievements_cache)} users")
        
        # Формируем задачи
        tasks = []
        skipped_users = 0
        
        for steamid, games_list in valid_users.items():
            steamid_str = str(steamid)
            
            games_with_2w = [g for g in games_list if g.get('playtime_2weeks', 0) > 0]
            games_with_2w_sorted = sorted(games_with_2w, key=lambda x: x.get('playtime_2weeks', 0), reverse=True)
            other_games = [g for g in games_list if g.get('playtime_2weeks', 0) == 0 and g.get('playtime_forever', 0) >= Config.MIN_PLAYTIME_MINUTES]
            other_games_sorted = sorted(other_games, key=lambda x: x.get('playtime_forever', 0), reverse=True)
            
            selected = games_with_2w_sorted[:10] + other_games_sorted[:10]
            if len(selected) < 3:
                all_sorted = sorted(games_list, key=lambda x: x.get('playtime_forever', 0), reverse=True)
                for g in all_sorted:
                    if g not in selected and len(selected) < 15 and g.get('playtime_forever', 0) >= Config.MIN_PLAYTIME_MINUTES:
                        selected.append(g)
            
            if steamid_str not in achievements_cache:
                achievements_cache[steamid_str] = {}
            
            missing_appids = []
            for game in selected:
                appid = game.get('appid')
                if not appid:
                    continue
                appid_str = str(appid)
                if appid_str not in achievements_cache[steamid_str]:
                    missing_appids.append(appid_str)
            
            if missing_appids:
                for appid_str in missing_appids:
                    tasks.append((steamid_str, appid_str))
            else:
                skipped_users += 1
        
        print(f"Total tasks: {len(tasks)}")
        print(f"Skipped users: {skipped_users}")
        
        if not tasks:
            print('All achievements already collected!')
        else:
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0'})
            
            processed = 0
            failed = 0
            saved_at = 0
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(get_achievements, api_key, sid, aid, session): (sid, aid) for sid, aid in tasks}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc='Collecting achievements'):
                    steamid, appid = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            if steamid not in achievements_cache:
                                achievements_cache[steamid] = {}
                            achievements_cache[steamid][appid] = {
                                'achievements': result,
                                'no_data': False
                            }
                            checked_users[steamid] = True
                        else:
                            if steamid not in achievements_cache:
                                achievements_cache[steamid] = {}
                            achievements_cache[steamid][appid] = {'no_data': True}
                            failed += 1
                    except Exception:
                        failed += 1
                    processed += 1
                    
                    if processed - saved_at >= Config.CHECKPOINT_INTERVAL:
                        save_json(achievements_cache, 'db/achievements', 'achievements')
                        save_json(checked_users, 'db', 'checked_users')
                        saved_at = processed
                        print(f'\nAuto-save after {processed} tasks')
            
            save_json(achievements_cache, 'db/achievements', 'achievements')
            save_json(checked_users, 'db', 'checked_users')
            
            print(f"Successfully loaded: {processed - failed} tasks")
            print(f"Failed: {failed} tasks")
        
        print(f'Step 4 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    # ==========================================
    # ШАГ 5: ДЕТАЛИ ИГР (скрапинг)
    # ==========================================
    if start_step <= 5 and not args.skip_game_details:
        print('\n' + '=' * 70)
        print('STEP 5: Scraping game details')
        print('=' * 70)
        step_start = time.time()
        
        game_details = load_all_games_details()
        existing_appids = set(game_details.keys())
        missing_appids = list(unique_appids - existing_appids)
        
        print(f"Existing: {len(existing_appids)} games")
        print(f"Need to collect: {len(missing_appids)} new games")
        
        if not missing_appids:
            print('All game details already collected!')
        else:
            missing_appids_int = [int(appid) for appid in missing_appids[:5000]]  # Ограничиваем для безопасности
            
            print(f'Processing {len(missing_appids_int)} games...')
            
            try:
                new_games_list, failed_games = scrape_games_batch(
                    missing_appids_int,
                    max_workers=Config.MAX_WORKERS
                )
                
                new_games_dict = convert_games_list_to_dict(new_games_list)
                game_details.update(new_games_dict)
                print(f"Total in database: {len(game_details)} games")
                
                save_json(game_details, 'db/games_detail', 'games_detail')
                
                if failed_games:
                    save_json(failed_games, 'db/failed_games', 'failed_games')
                    print(f"Failed to collect: {len(failed_games)} games")
            except Exception as e:
                print(f"Error during scraping: {e}")
                traceback.print_exc()
        
        print(f'Step 5 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    # ==========================================
    # ШАГ 6: АНАЛИЗ СТИЛЕЙ
    # ==========================================
    if start_step <= 6 and not args.skip_style_analysis:
        print('\n' + '=' * 70)
        print('STEP 6: Gameplay style analysis')
        print('=' * 70)
        step_start = time.time()
        
        if not api_key:
            print("ERROR: API_KEY required for this step")
            return
        
        games_data = load_latest_json('db/games', 'games')
        if not games_data:
            print('ERROR: No games data')
            return
        
        games_detail = load_all_games_details()
        achievements_cache = load_latest_json('db/achievements', 'achievements') or {}
        checked_users = load_latest_json('db', 'checked_users') or {}
        
        valid_users = {sid: glist for sid, glist in games_data.items() if is_valid_profile(glist)}
        print(f"Users with games: {len(valid_users)}")
        print(f"Achievement cache: {len(achievements_cache)} users")
        
        analyzer = StyleAnalyzer(api_key)
        profiles = {}
        processed = 0
        
        print('\nProcessing users...')
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(analyzer.process_user, sid, glist, games_detail, checked_users, achievements_cache): sid
                for sid, glist in valid_users.items()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc='Processing users'):
                sid = futures[future]
                try:
                    profile = future.result()
                    if profile:
                        profiles[sid] = profile
                except Exception as e:
                    print(f'Error processing {sid}: {e}')
                processed += 1
                
                if processed % Config.CHECKPOINT_INTERVAL == 0:
                    save_json(profiles, 'db/play_style_profiles', 'play_style_profiles', is_checkpoint=True)
        
        final_file = save_json(profiles, 'db/play_style_profiles', 'play_style_profiles')
        
        # Статистика
        total_users = len(profiles)
        users_with_style = sum(1 for p in profiles.values() if p['dominant_style'] != 'unknown')
        users_with_ach = sum(1 for p in profiles.values() if p['games_with_achievements'] > 0)
        
        print('\n' + '=' * 60)
        print('STYLE ANALYSIS RESULTS')
        print('=' * 60)
        print(f'Total users: {total_users}')
        print(f'With style: {users_with_style} ({users_with_style/total_users*100:.1f}%)')
        print(f'With achievements: {users_with_ach} ({users_with_ach/total_users*100:.1f}%)')
        
        style_counts = {s: 0 for s in STYLES + ['unknown']}
        for p in profiles.values():
            style_counts[p['dominant_style']] += 1
        
        print('\nStyle distribution:')
        for style, count in style_counts.items():
            pct = count / total_users * 100 if total_users > 0 else 0
            bar = '█' * int(pct / 2)
            print(f'  {style:15} | {bar:30} | {count:6} ({pct:.1f}%)')
        
        print(f'\nFinal file: {final_file}')
        print(f'Step 6 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    # ==========================================
    # ШАГ 7: АНАЛИЗ ТРЕНДОВ
    # ==========================================
    if not args.skip_trends:
        print('\n' + '=' * 70)
        print('STEP 7: Trend analysis and segmentation')
        print('=' * 70)
        step_start = time.time()
        
        profiles = load_latest_json('db/play_style_profiles', 'play_style_profiles')
        if not profiles:
            print('WARNING: No profiles found, skipping trend analysis')
        else:
            games_detail = load_all_games_details()
            games_list = load_all_games_list()
            games_name_map = build_games_name_map(games_list) if games_list else {}
            
            print(f'Loaded profiles: {len(profiles)}')
            print(f'Game names: {len(games_name_map)}')
            
            total_users = len(profiles)
            
            # Сравнение стилей
            style_changes = []
            style_changed_count = 0
            change_matrix = defaultdict(lambda: defaultdict(int))
            
            for steamid, profile in profiles.items():
                style_all = profile.get('dominant_style', 'unknown')
                style_2w = profile.get('dominant_style_2w', 'none')
                
                if style_2w == 'none' or style_all == 'unknown':
                    continue
                
                changed = style_all != style_2w
                change_matrix[style_all][style_2w] += 1
                
                if changed:
                    style_changed_count += 1
                    style_changes.append({
                        'steamid': steamid,
                        'from_style': style_all,
                        'to_style': style_2w,
                        'playtime_2w_hours': profile.get('playtime_2weeks_hours', 0)
                    })
            
            active_users = sum(1 for p in profiles.values() if p.get('dominant_style_2w') != 'none')
            
            # Сегментация
            segments = {seg: [] for seg in Config.SEGMENT_THRESHOLDS.keys()}
            
            for steamid, profile in profiles.items():
                hours = profile.get('playtime_2weeks_hours', 0)
                segment = get_segment(hours)
                if segment in segments:
                    segments[segment].append(profile)
            
            # Топ игры
            top_games_2w = Counter()
            game_playtime_2w = defaultdict(int)
            
            if games_list:
                for steamid, user_games in games_list.items():
                    if not isinstance(user_games, list):
                        continue
                    for game in user_games:
                        playtime_2w = game.get('playtime_2weeks', 0)
                        if playtime_2w > 0:
                            appid = str(game.get('appid'))
                            top_games_2w[appid] += 1
                            game_playtime_2w[appid] += playtime_2w
            
            # Топ-30
            top_30_games = []
            for appid, players_count in top_games_2w.most_common(30):
                total_minutes = game_playtime_2w[appid]
                game_name = get_game_name(games_name_map, games_detail, appid)
                top_30_games.append({
                    'appid': appid,
                    'name': game_name,
                    'players_count': players_count,
                    'total_playtime_2w_hours': round(total_minutes / 60, 1),
                    'avg_playtime_per_player_hours': round(total_minutes / 60 / players_count, 1) if players_count > 0 else 0,
                })
            
            # Сохранение результатов
            analysis_result = {
                'generated_at': datetime.now().isoformat(),
                'total_users': total_users,
                'active_users_2w': active_users,
                'style_comparison': {
                    'users_with_style_change': style_changed_count,
                    'change_percentage': round(style_changed_count / active_users * 100, 1) if active_users > 0 else 0,
                    'change_matrix': {k: dict(v) for k, v in change_matrix.items()},
                },
                'segments': {
                    seg_name: {
                        'total_users': len(seg_profiles),
                        'percentage': round(len(seg_profiles) / total_users * 100, 1),
                        'avg_playtime_2w_hours': round(sum(p.get('playtime_2weeks_hours', 0) for p in seg_profiles) / len(seg_profiles), 2) if seg_profiles else 0,
                    }
                    for seg_name, seg_profiles in segments.items()
                },
                'top_games_2w': top_30_games,
            }
            
            result_file = save_json(analysis_result, 'db/play_style_trends', 'play_style_trends')
            
            # Вывод
            print('\n' + '=' * 60)
            print('TREND ANALYSIS RESULTS')
            print('=' * 60)
            print(f'Active users: {active_users}')
            change_pct = round(style_changed_count / active_users * 100, 1) if active_users > 0 else 0
            print(f'Style changed: {style_changed_count} ({change_pct}%)')
            
            print('\nSegmentation:')
            for seg_name in ['hardcore', 'regular', 'casual', 'light', 'inactive']:
                seg_data = analysis_result['segments'].get(seg_name, {})
                total = seg_data.get('total_users', 0)
                pct = seg_data.get('percentage', 0)
                avg_h = seg_data.get('avg_playtime_2w_hours', 0)
                bar = '█' * int(pct / 2)
                print(f'  {seg_name.upper():10} | {bar:30} | {total:6} ({pct:.1f}%) | {avg_h:.1f}h avg')
            
            print(f'\nTop 5 games (2 weeks):')
            for i, game in enumerate(top_30_games[:5], 1):
                print(f'  {i}. {game["name"][:40]:40} | {game["players_count"]:>6} players')
            
            print(f'\nResults saved: {result_file}')
        
        print(f'Step 7 completed in {(time.time() - step_start) / 60:.2f} min.')
    
    # ==========================================
    # ИТОГ
    # ==========================================
    total_time = (time.time() - total_start_time) / 60
    print('\n' + '=' * 70)
    print(f'ALL STEPS COMPLETED IN {total_time:.2f} MINUTES!')
    print('=' * 70)
    
    print('\nFINAL STATISTICS:')
    print(f"  - Unique SteamIDs: {len(all_target_steamids)}")
    print(f"  - Users in database: {len(games_data) if games_data else 0}")
    print(f"  - Open profiles: {sum(1 for g in (games_data or {}).values() if is_valid_profile(g))}")
    print(f"  - User details: {len(user_details) if user_details else 0}")
    print(f"  - Game details: {len(game_details) if game_details else 0}")
    print(f"  - Unique appids: {len(unique_appids)}")
    print(f"  - Achievements cache: {len(achievements_cache) if achievements_cache else 0} users")
    print(f"  - Style profiles: {len(profiles) if 'profiles' in locals() else 0}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    except Exception as e:
        print(f'\nCritical error: {e}')
        traceback.print_exc()