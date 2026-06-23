# %%


import os
import sys
import glob
import json
import time
import random
import traceback
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# ==========================================
# НАСТРОЙКА ПУТЕЙ (работает из любой директории)
# ==========================================
# Папка, где лежит этот скрипт
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Корень проекта (на уровень выше, если скрипт в подпапке)
# Или та же папка, если скрипт в корне
if os.path.basename(SCRIPT_DIR) in ['1. data_collection', 'data_collection', 'src']:
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
else:
    PROJECT_ROOT = SCRIPT_DIR

# Меняем рабочую директорию на папку скрипта
os.chdir(SCRIPT_DIR)

# Добавляем папку скрипта в PYTHONPATH для импортов
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ==========================================
# ДИАГНОСТИКА ПРИ ЗАПУСКЕ
# ==========================================
print('=' * 70)
print('UNIFIED STEAM DATA COLLECTOR')
print('=' * 70)
print(f'[DEBUG] Скрипт находится в: {SCRIPT_DIR}')
print(f'[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}')
print(f'[DEBUG] Текущая рабочая директория: {os.getcwd()}')
print(f'[DEBUG] Python version: {sys.version}')
print('=' * 70)

# ==========================================
# ИМПОРТЫ (с обработкой ошибок)
# ==========================================
try:
    from steam_scraper import users, games, users_detail, games_detail
    print('[OK] Модули steam_scraper успешно импортированы')
except ImportError as e:
    print(f'[ERROR] Не удалось импортировать steam_scraper: {e}')
    print(f'[HINT] Проверьте, что папка steam_scraper существует в: {SCRIPT_DIR}')
    print(f'[HINT] Содержимое папки: {os.listdir(SCRIPT_DIR)}')
    sys.exit(1)

# ==========================================
# ЗАГРУЗКА .env (из правильной директории)
# ==========================================
env_path = os.path.join(PROJECT_ROOT, '.env')
if not os.path.exists(env_path):
    env_path = os.path.join(SCRIPT_DIR, '.env')

if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f'[OK] Загружен .env из: {env_path}')
else:
    print(f'[WARN] Файл .env не найден ни в {PROJECT_ROOT}, ни в {SCRIPT_DIR}')
    print('[WARN] API_KEY может быть недоступен')

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
TARGET_OPEN_PROFILES = 19_384
CHECKPOINT_INTERVAL = 2000
BATCH_SIZE = 100
MAX_WORKERS = 6


# ==========================================
# УТИЛИТЫ
# ==========================================
def load_json_if_exists(directory, prefix):
    """Загрузка JSON с приоритетом контрольной точки"""
    full_path = os.path.join(PROJECT_ROOT, directory)
    checkpoint_file = os.path.join(full_path, f'{prefix}_checkpoint.json')
    
    if os.path.exists(checkpoint_file):
        print(f"[INFO] Найдена контрольная точка: {checkpoint_file}")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f), checkpoint_file
    
    if not os.path.exists(full_path):
        return None, None
    
    files = glob.glob(os.path.join(full_path, f'{prefix}_*.json'))
    if not files:
        return None, None
    
    files.sort(key=os.path.getmtime, reverse=True)
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[INFO] Загружен файл: {os.path.basename(filepath)}")
                return data, filepath
        except json.JSONDecodeError:
            continue
    
    return None, None


def save_json(data, directory, prefix, is_checkpoint=False):
    """Сохранение JSON"""
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
        print(f"[SAVE] Сохранено: {filepath}")
    return filepath


def get_ids_list(data):
    """Безопасное получение списка ID"""
    if isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, list):
        return data
    return []


def is_valid_profile(games_list):
    """Проверка: профиль открытый и содержит игры"""
    if not isinstance(games_list, list) or len(games_list) == 0:
        return False
    if len(games_list) == 1 and isinstance(games_list[0], dict) and games_list[0].get('private'):
        return False
    return True


def load_all_games_details():
    """Загружает ВСЕ файлы games_detail из папки db/games_detail."""
    full_path = os.path.join(PROJECT_ROOT, 'db/games_detail')
    if not os.path.exists(full_path):
        return {}
    
    combined = {}
    files = glob.glob(os.path.join(full_path, '*.json'))
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        appid = item.get('gameid') or item.get('appid') or item.get('id')
                        if appid:
                            combined[str(appid)] = item
            elif isinstance(data, dict):
                for k, v in data.items():
                    combined[str(k)] = v
        except Exception as e:
            print(f"[WARN] Ошибка загрузки {os.path.basename(filepath)}: {e}")
            continue
    
    return combined


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
# ОСНОВНАЯ ФУНКЦИЯ
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Steam Data Collector (Unified)')
    parser.add_argument('--force-step', type=int, choices=[1, 2, 3, 4],
                        help='Принудительно начать с указанного шага')
    parser.add_argument('--skip-step-4', action='store_true',
                        help='Пропустить шаг 4 (скрапинг игр)')
    args = parser.parse_args()
    
    # Проверка API_KEY
    api_key = os.getenv('API_KEY')
    if not api_key:
        print('[ERROR] API_KEY не найден в .env!')
        print(f'[HINT] Проверьте файл .env в: {PROJECT_ROOT}')
        response = input('Продолжить без API_KEY? (y/n): ')
        if response.lower() != 'y':
            return
    
    total_start_time = time.time()
    
    print('\n' + '=' * 70)
    print('STARTING DATA COLLECTION')
    print('=' * 70)
    
    # ==========================================
    # АНАЛИЗ ТЕКУЩЕГО СОСТОЯНИЯ
    # ==========================================
    print('\n[INFO] Анализ текущего состояния базы...')
    
    friends_data, _ = load_json_if_exists('db/users', 'users')
    unique_ids_data, _ = load_json_if_exists('db/unique_ids', 'unique_ids')
    games_data, _ = load_json_if_exists('db/games', 'games')
    user_details, _ = load_json_if_exists('db/users_detail', 'users_detail')
    game_details = load_all_games_details()
    
    print(f"\n[STAT] Текущее состояние:")
    print(f"  - friends_data: {'есть' if friends_data else 'нет'}")
    print(f"  - unique_ids_data: {'есть' if unique_ids_data else 'нет'}")
    print(f"  - games_data: {'есть' if games_data else 'нет'}")
    print(f"  - user_details: {'есть' if user_details else 'нет'}")
    print(f"  - game_details: {len(game_details)} записей")
    
    # Определяем стартовый шаг
    start_step = 1
    if args.force_step:
        start_step = args.force_step
        print(f'\n[WARN] Принудительный старт с шага {start_step}')
    else:
        if friends_data and unique_ids_data:
            start_step = max(start_step, 2)
        if games_data and isinstance(games_data, dict):
            valid_count = sum(1 for g in games_data.values() if is_valid_profile(g))
            if valid_count >= TARGET_OPEN_PROFILES:
                start_step = max(start_step, 3)
        if user_details and isinstance(user_details, dict) and games_data:
            valid_steamids = {sid for sid, g in games_data.items() if is_valid_profile(g)}
            if len(user_details) >= len(valid_steamids) * 0.95:
                start_step = max(start_step, 4)
        
        print(f'\n[INFO] Автоматический старт с шага {start_step}')
    
    # ==========================================
    # ШАГ 1: СБОР ДРУЗЕЙ И УНИКАЛЬНЫХ ID
    # ==========================================
    if start_step <= 1:
        print('\n' + '=' * 70)
        print('[STEP 1] Сбор друзей и уникальных SteamID')
        print('=' * 70)
        step_start = time.time()
        
        if friends_data is None or unique_ids_data is None:
            print('[WARN] Данные не найдены. Запускаем полный сбор...')
            friends_data, unique_steamids = users.fetch_friend_data(
                api_key=os.getenv('API_KEY'),
                initial_steamid='76561198147702705',
                max_unique_ids=260_000
            )
            save_json(friends_data, 'db/users', 'users')
            save_json(unique_steamids, 'db/unique_ids', 'unique_ids')
        else:
            unique_steamids = unique_ids_data
            print(f'[OK] Данные уже собраны ({len(get_ids_list(unique_steamids))} профилей)')
        
        print(f'[TIME] Шаг 1: {(time.time() - step_start) / 60:.2f} мин.')
    
    all_target_steamids = get_ids_list(unique_ids_data) if unique_ids_data else []
    all_target_steamids_set = set(all_target_steamids)
    
    # ==========================================
    # ШАГ 2: СБОР ИГР
    # ==========================================
    if start_step <= 2:
        print('\n' + '=' * 70)
        print('[STEP 2] Сбор игр пользователей')
        print('=' * 70)
        step_start = time.time()
        
        if games_data is None or not isinstance(games_data, dict):
            games_data = {}
        
        current_valid = sum(1 for g in games_data.values() if is_valid_profile(g))
        remaining = max(0, TARGET_OPEN_PROFILES - current_valid)
        
        print(f"[STAT] В базе: {current_valid} открытых профилей")
        print(f"[STAT] Цель: {TARGET_OPEN_PROFILES}. Осталось: {remaining}")
        
        existing_ids = set(games_data.keys())
        missing_ids = list(all_target_steamids_set - existing_ids)
        
        if remaining == 0:
            print('[OK] Цель достигнута! Шаг 2 пропускается.')
        elif not missing_ids:
            print('[OK] Все пользователи уже проверены.')
        else:
            print(f"[INFO] Проверяем {len(missing_ids)} новых пользователей...")
            
            def fetch_games(steamid, max_retries=3):
                for attempt in range(max_retries):
                    try:
                        response = games.get_games_data(
                            key=os.getenv('API_KEY'),
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
            
            processed = 0
            new_valid = 0
            private_count = 0
            goal_reached = False
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_games, sid): sid for sid in missing_ids}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc='Checking profiles'):
                    steamid, user_games = future.result()
                    games_data[steamid] = user_games
                    processed += 1
                    
                    if is_valid_profile(user_games):
                        new_valid += 1
                    elif user_games and user_games[0].get('private'):
                        private_count += 1
                    
                    total_valid_now = current_valid + new_valid
                    
                    if total_valid_now >= TARGET_OPEN_PROFILES and not goal_reached:
                        print(f"\n[TARGET] Цель достигнута! Открытых: {total_valid_now}, Приватных: {private_count}")
                        goal_reached = True
                        save_json(games_data, 'db/games', 'games')
                        break
                    
                    if processed % CHECKPOINT_INTERVAL == 0 and not goal_reached:
                        save_json(games_data, 'db/games', 'games', is_checkpoint=True)
            
            if not goal_reached:
                save_json(games_data, 'db/games', 'games')
            
            print(f"[STAT] Итог шага 2: +{new_valid} открытых, {private_count} приватных")
        
        print(f'[TIME] Шаг 2: {(time.time() - step_start) / 60:.2f} мин.')
    
    # Извлекаем уникальные appid
    unique_appids = set()
    for games_list in games_data.values():
        if is_valid_profile(games_list):
            for game in games_list:
                if isinstance(game, dict) and 'appid' in game:
                    unique_appids.add(str(game['appid']))
    
    print(f"[STAT] Всего уникальных appid в профилях: {len(unique_appids)}")
    
    # ==========================================
    # ШАГ 3: ДЕТАЛИ ПОЛЬЗОВАТЕЛЕЙ
    # ==========================================
    if start_step <= 3:
        print('\n' + '=' * 70)
        print('[STEP 3] Сбор деталей пользователей')
        print('=' * 70)
        step_start = time.time()
        
        if user_details is None or not isinstance(user_details, dict):
            user_details = {}
        
        existing_ids = set(user_details.keys())
        valid_steamids = {sid for sid, g in games_data.items() if is_valid_profile(g)}
        missing_ids = list(valid_steamids - existing_ids)
        
        if not missing_ids:
            print(f'[OK] Детали уже собраны для всех {len(valid_steamids)} пользователей.')
        else:
            print(f'[INFO] Собираем детали для {len(missing_ids)} новых пользователей...')
            
            def fetch_user_detail(steamid, max_retries=3):
                for attempt in range(max_retries):
                    try:
                        response = users_detail.get_users_detail(
                            key=os.getenv('API_KEY'),
                            steamids=steamid
                        )
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
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(fetch_user_detail, sid): sid for sid in missing_ids}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc='Fetching user details'):
                    steamid, player_data = future.result()
                    if player_data:
                        user_details[steamid] = player_data
            
            save_json(user_details, 'db/users_detail', 'users_detail')
        
        print(f'[TIME] Шаг 3: {(time.time() - step_start) / 60:.2f} мин.')
    
    # ==========================================
    # ШАГ 4: СКРАПИНГ ДЕТАЛЕЙ ИГР
    # ==========================================
    if start_step <= 4 and not args.skip_step_4:
        print('\n' + '=' * 70)
        print('[STEP 4] Скрапинг деталей игр (Steam API)')
        print('=' * 70)
        step_start = time.time()
        
        existing_appids = set(game_details.keys())
        missing_appids = list(unique_appids - existing_appids)
        
        print(f"[STAT] В базе games_detail: {len(existing_appids)} игр")
        print(f"[STAT] Нужно собрать: {len(missing_appids)} новых игр")
        
        if not missing_appids:
            print('[OK] Все детали игр уже собраны!')
        else:
            missing_appids_int = [int(appid) for appid in missing_appids]
            
            print(f'[INFO] Запускаем блочный скрапинг {len(missing_appids_int)} игр...')
            
            try:
                new_games_list, failed_games = games_detail.scrape_games_in_batches(
                    missing_appids_int,
                    batch_size=BATCH_SIZE,
                    max_workers=MAX_WORKERS
                )
                
                print(f"\n[INFO] Конвертация результатов (было {len(new_games_list)} записей в списке)...")
                new_games_dict = convert_games_list_to_dict(new_games_list)
                print(f"[OK] Получен словарь из {len(new_games_dict)} записей")
                
                game_details.update(new_games_dict)
                print(f"[STAT] Итого в базе: {len(game_details)} игр")
                
                save_json(game_details, 'db/games_detail', 'games_detail')
                
                if failed_games:
                    save_json(failed_games, 'db/failed_games', 'failed_games')
                    print(f"[WARN] Не удалось собрать: {len(failed_games)} игр")
            except Exception as e:
                print(f"[ERROR] Ошибка при скрапинге: {e}")
                traceback.print_exc()
        
        print(f'[TIME] Шаг 4: {(time.time() - step_start) / 60:.2f} мин.')
    
    # ==========================================
    # ИТОГ
    # ==========================================
    total_time = (time.time() - total_start_time) / 60
    print('\n' + '=' * 70)
    print(f'[DONE] Все шаги завершены за {total_time:.2f} мин.!')
    print('=' * 70)
    
    print('\n[FINAL STATISTICS]:')
    print(f"  - Уникальных SteamID: {len(all_target_steamids)}")
    print(f"  - Пользователей в базе: {len(games_data) if games_data else 0}")
    print(f"  - Открытых профилей: {sum(1 for g in (games_data or {}).values() if is_valid_profile(g))}")
    print(f"  - Деталей пользователей: {len(user_details) if user_details else 0}")
    print(f"  - Деталей игр: {len(game_details)}")
    print(f"  - Уникальных appid: {len(unique_appids)}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[WARN] Прервано пользователем')
    except Exception as e:
        print(f'\n[ERROR] Критическая ошибка: {e}')
        traceback.print_exc()