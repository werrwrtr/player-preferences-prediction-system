# This script processes the data collected by the steam_scraper script and creates a database of users and their games.
# %% 
import os
import glob
import json
import datetime
import pandas as pd
import re

# КОРЕНЬ ПРОЕКТА
main_folder = r"D:\steam_recommender_system"

# Папка db, которая лежит В КОРНЕ проекта
db_root_folder = os.path.join(main_folder, 'db')

# Папка второго этапа, куда мы сохраним результаты
output_folder = os.path.join(main_folder, '2. data_processing', 'db')

# Function to save full DataFrame to parquet
def save_parquet_full(data):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    df = pd.DataFrame(data)
    out_dir = os.path.join(output_folder, 'full')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{now}.parquet')
    df.to_parquet(path, index=False)
    print(f"Полный датасет успешно сохранен в: {path}")

# Function to save filtered DataFrame to parquet
def save_parquet_filtered(data):
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    df = pd.DataFrame(data)
    out_dir = os.path.join(output_folder, 'filtered')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{now}.parquet')
    df.to_parquet(path, index=False)
    print(f"Фильтрованный датасет успешно сохранен в: {path}")

# 1. Load users_detail
print("Загрузка users_detail...")
users_detail_folder = os.path.join(db_root_folder, 'users_detail')
users_detail_files = glob.glob(os.path.join(users_detail_folder, '*.json'))

if users_detail_files:
    latest_users_detail_file = max(users_detail_files, key=os.path.getctime)
    print(f"  └─ Используем файл пользователей: {os.path.basename(latest_users_detail_file)}")
    with open(latest_users_detail_file, 'r', encoding='utf-8') as file:
        users_detail_json = json.load(file)
    
    # Если это словарь {steamid: {данные}}, аккуратно превращаем в список
    if isinstance(users_detail_json, dict):
        for k, v in users_detail_json.items():
            if isinstance(v, dict) and 'steamid' not in v:
                v['steamid'] = k
        users_detail_list = list(users_detail_json.values())
    else:
        users_detail_list = users_detail_json

    # Создаем датафрейм с гарантированно уникальными именами колонок
    users_detail = pd.DataFrame(users_detail_list)
    users_detail = users_detail.loc[:, ~users_detail.columns.duplicated()]
    
    # Очищаем строки от дубликатов по ID, если они есть
    if 'steamid' in users_detail.columns:
        users_detail.drop_duplicates(subset=['steamid'], keep='first', inplace=True)
else:
    print(f'Ошибка: JSON-файлы не найдены в папке: {users_detail_folder}')
    exit()

# 2. Load games
print("\nЗагрузка базы игр пользователей...")
games_folder = os.path.join(db_root_folder, 'games')
games_files = glob.glob(os.path.join(games_folder, '*.json'))

if games_files:
    latest_games_file = max(games_files, key=os.path.getctime)
    print(f"  └─ Используем файл игр пользователей: {os.path.basename(latest_games_file)}")
    with open(latest_games_file, 'r', encoding='utf-8') as file:
        games_json = json.load(file)
    rows = []
    for steam_id, user_games in games_json.items():
        if user_games:
            if len(user_games) == 1 and user_games[0].get('private'):
                continue
            for game in user_games:
                if isinstance(game, dict):
                    row = {'steam_id': steam_id, **game}
                    rows.append(row)
        else:
            rows.append({'steam_id': steam_id})
    games = pd.DataFrame(rows)
else:
    print(f'Ошибка: JSON-файлы не найдены в папке: {games_folder}')
    exit()

# 3. Load games_detail from files with the same date
print("\nЗагрузка и склейка метаданных игр (games_detail)...")
games_detail_folder = os.path.join(db_root_folder, 'games_detail')
games_detail_files = glob.glob(os.path.join(games_detail_folder, '*.json'))

if games_detail_files:
    latest_file = max(games_detail_files, key=os.path.getctime)
    latest_file_date = re.search(r'\d{4}-\d{2}-\d{2}', latest_file).group()
    print(f"  └─ Найдена дата последних обновлений: {latest_file_date}")

    same_date_files = [f for f in games_detail_files if latest_file_date in f]
    print(f"  └─ Найдено файлов/батчей для склейки: {len(same_date_files)}")

    games_detail_list = []
    for file_path in same_date_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                games_detail_json = json.load(file)
                if isinstance(games_detail_json, dict):
                    games_detail_json = list(games_detail_json.values())
                
                if games_detail_json:
                    games_detail_list.append(pd.DataFrame(games_detail_json))
        except Exception:
            continue

    if games_detail_list:
        games_detail = pd.concat(games_detail_list, ignore_index=True)
    else:
        print('Ошибка: Не удалось извлечь данные из файлов метаданных игр.')
        exit()
else:
    print(f'Ошибка: JSON-файлы не найдены в папке: {games_detail_folder}')
    exit()

# Очистка данных
if 'err' in games_detail.columns:
    games_detail = games_detail[games_detail['err'].isna()]
games_detail.dropna(subset=['gameid'], inplace=True)
games_detail.drop_duplicates(subset='gameid', inplace=True)
games_detail['gameid'] = pd.to_numeric(games_detail['gameid'], errors='coerce').astype('int64')

# Добавляем префиксы
users_detail_prefixed = users_detail.add_prefix('user_')
games_prefixed = games.add_prefix('game_')
games_detail_prefixed = games_detail.add_prefix('game_detail_')

# 4. Merge DataFrames
print("\nОбъединение таблиц (pd.merge)...")
merged_users_games = pd.merge(
    users_detail_prefixed,
    games_prefixed, 
    how='left', 
    left_on='user_steamid', 
    right_on='game_steam_id'
)

merged_users_games['game_appid'] = (
    merged_users_games['game_appid']
    .fillna(0)
    .astype('int64')
)

merged_users_games_details = pd.merge(
    merged_users_games,
    games_detail_prefixed,
    how='left',
    left_on='game_appid',
    right_on='game_detail_gameid'
)

# Сохранение результатов
print("\nСохранение результатов...")
save_parquet_full(merged_users_games_details)

filtered_columns = [
    'user_steamid', 'user_personaname', 'user_loccountrycode',
    'game_appid', 'game_name', 'game_playtime_forever',
    'game_detail_release_date', 'game_detail_developer', 'game_detail_publisher',
    'game_detail_genres', 'game_detail_tags', 'game_detail_rating'
]

existing_filtered_cols = [col for col in filtered_columns if col in merged_users_games_details.columns]

if 'game_img_icon_url' in merged_users_games_details.columns:
    existing_filtered_cols.append('game_img_icon_url')

filtered_df = merged_users_games_details[existing_filtered_cols]
save_parquet_filtered(filtered_df)

print("\n🎉 [УСПЕХ] Этап 2 успешно выполнен! Все данные объединены.")
# %%
