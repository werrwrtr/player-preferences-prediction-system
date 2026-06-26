# Web Application for Steam Recommendation System with Cold Start Support
import os
import glob
import re
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import dill as pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Сброс кэша при перезапусках
st.cache_resource.clear()

# ==============================================================================
# 1. КОНФИГУРАЦИЯ СТРАНИЦЫ
# ==============================================================================
st.set_page_config(
    page_title='Steam Recommender System',
    page_icon='🎮',
    layout='wide'
)

st.markdown("""
    <style>
        .stApp {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', sans-serif;
        }
        .user-profile {
            background: linear-gradient(145deg, #1e1e2f, #2a2a3e);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #3a3a5c;
        }
    </style>
""", unsafe_allow_html=True)

# Пути
current_dir = os.getcwd()
if "predictions" in current_dir or "recommender_system" in current_dir:
    base_model_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '4. model'))
else:
    base_model_dir = os.path.abspath(current_dir)

if not os.path.exists(base_model_dir) or not os.path.exists(os.path.join(base_model_dir, 'hft_hybrid_model.pkl')):
    base_model_dir = os.path.abspath(current_dir)

base_db_dir = os.path.join(base_model_dir, 'db')

# Steam API ключ
STEAM_API_KEY = '*'

# ==============================================================================
# 2. STEAM API ФУНКЦИИ
# ==============================================================================
@st.cache_data(ttl=3600)
def resolve_vanity_url(vanity_url):
    """
    Конвертирует vanity URL (например, 'sukadedins1241') в SteamID64.
    """
    url = "http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/"
    params = {
        'key': STEAM_API_KEY,
        'vanityurl': vanity_url,
        'url_type': 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('response', {}).get('success') == 1:
            return data['response'].get('steamid')
    except Exception as e:
        print(f"[ERROR] ResolveVanityURL: {e}")
    
    return None


def parse_steam_input(input_text):
    """
    Определяет тип ввода и извлекает SteamID64.
    Поддерживает:
    - 17-значный SteamID64: 76561198999999999
    - Vanity URL: steamcommunity.com/id/sukadedins1241
    - Профиль URL: steamcommunity.com/profiles/76561198999999999
    - Просто имя: sukadedins1241
    """
    input_text = input_text.strip()
    
    # Паттерн 1: 17-значный SteamID64
    if re.match(r'^\d{17}$', input_text):
        return input_text
    
    # Паттерн 2: URL с vanity (steamcommunity.com/id/NAME)
    vanity_match = re.search(r'steamcommunity\.com/id/([^/\s]+)', input_text)
    if vanity_match:
        vanity_name = vanity_match.group(1)
        steam_id = resolve_vanity_url(vanity_name)
        return steam_id
    
    # Паттерн 3: URL с profiles (steamcommunity.com/profiles/STEAMID)
    profile_match = re.search(r'steamcommunity\.com/profiles/(\d{17})', input_text)
    if profile_match:
        return profile_match.group(1)
    
    # Паттерн 4: Просто имя (vanity URL без домена)
    if re.match(r'^[a-zA-Z0-9_\-]+$', input_text) and len(input_text) < 32:
        steam_id = resolve_vanity_url(input_text)
        return steam_id
    
    return None


@st.cache_data(ttl=3600)
def get_player_summary(steam_id):
    """Получает информацию о пользователе через Steam API."""
    url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    params = {
        'key': STEAM_API_KEY,
        'steamids': steam_id,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'response' in data and 'players' in data['response'] and data['response']['players']:
            player = data['response']['players'][0]
            return {
                'personaname': player.get('personaname', 'Unknown'),
                'avatar': player.get('avatarfull', ''),
                'profileurl': player.get('profileurl', ''),
                'loccountrycode': player.get('loccountrycode', ''),
                'personastate': player.get('personastate', 0),
            }
    except Exception as e:
        print(f"[ERROR] GetPlayerSummaries: {e}")
    
    return None


@st.cache_data(ttl=3600)
def get_recently_played_games(steam_id):
    """Получает игры, в которые пользователь играл за последние 2 недели."""
    url = "http://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v0001/"
    params = {
        'key': STEAM_API_KEY,
        'steamid': steam_id,
        'count': 50,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'response' in data and 'games' in data['response']:
            games = data['response']['games']
            return {
                'total_count': data['response'].get('total_count', len(games)),
                'games': [{
                    'appid': g['appid'],
                    'name': g.get('name', 'Unknown'),
                    'playtime_2weeks': g.get('playtime_2weeks', 0),
                    'playtime_forever': g.get('playtime_forever', 0),
                    'img_icon_url': g.get('img_icon_url', ''),
                } for g in games]
            }
    except Exception as e:
        print(f"[ERROR] GetRecentlyPlayedGames: {e}")
    
    return None


@st.cache_data(ttl=3600)
def get_game_details(app_id):
    """Получает детальную информацию об игре (жанры, теги, описание)."""
    url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&l=russian"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if str(app_id) in data and data[str(app_id)]['success']:
            game_data = data[str(app_id)]['data']
            
            genres = []
            if 'genres' in game_data:
                genres = [g['description'] for g in game_data['genres']]
            
            categories = []
            if 'categories' in game_data:
                categories = [c['description'] for c in game_data['categories']]
            
            return {
                'name': game_data.get('name', ''),
                'genres': genres,
                'categories': categories,
                'short_description': game_data.get('short_description', ''),
                'developers': game_data.get('developers', []),
                'publishers': game_data.get('publishers', []),
            }
    except Exception as e:
        pass
    
    return None


@st.cache_data(ttl=3600)
def get_owned_games_with_details(steam_id):
    """
    Получает ВСЕ игры пользователя через Steam API + детали каждой игры.
    """
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        'key': STEAM_API_KEY,
        'steamid': steam_id,
        'include_appinfo': True,
        'include_played_free_games': True,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'response' in data and 'games' in data['response']:
            games = data['response']['games']
            
            rated_list = []
            unrated_list = []
            games_details = []
            
            for game in games:
                app_id = game['appid']
                playtime = game.get('playtime_forever', 0)
                name = game.get('name', 'Unknown')
                
                # Получаем детали игры (жанры, теги)
                details = get_game_details(app_id)
                
                game_info = {
                    'game_appid': app_id,
                    'game_name': name,
                    'genres': details['genres'] if details else [],
                    'categories': details['categories'] if details else [],
                    'developers': details['developers'] if details else [],
                }
                games_details.append(game_info)
                
                if playtime > 120:  # > 2 часов
                    rating = min(5, max(1, int(playtime / 500) + 3))
                    rated_list.append({
                        'user_steamid': steam_id,
                        'game_appid': app_id,
                        'game_name': name,
                        'rating': rating,
                        'game_playtime_forever': playtime,
                        'genres': details['genres'] if details else [],
                    })
                else:
                    unrated_list.append({
                        'user_steamid': steam_id,
                        'game_appid': app_id,
                        'game_name': name,
                        'genres': details['genres'] if details else [],
                    })
            
            return {
                'game_count': len(games),
                'rated_df': pd.DataFrame(rated_list),
                'unrated_df': pd.DataFrame(unrated_list),
                'games_details': games_details
            }
    except Exception as e:
        print(f"[ERROR] GetOwnedGames: {e}")
    
    return None


def get_steam_user_data(steam_id):
    """Комплексная функция: получает все данные о пользователе."""
    result = {
        'profile': None,
        'recent_games': None,
        'owned_games': None,
        'error': None
    }
    
    result['profile'] = get_player_summary(steam_id)
    if not result['profile']:
        result['error'] = "❌ Не удалось получить информацию о профиле."
        return result
    
    result['recent_games'] = get_recently_played_games(steam_id)
    result['owned_games'] = get_owned_games_with_details(steam_id)
    
    if not result['owned_games']:
        result['error'] = "⚠️ Профиль открыт, но библиотека игр недоступна."
    
    return result


# ==============================================================================
# 3. КОНТЕНТНАЯ МОДЕЛЬ НА ЛЕТУ (для новых пользователей)
# ==============================================================================
def build_content_model_for_new_user(rated_df, unrated_df, games_details):
    """
    Строит TF-IDF профиль пользователя на основе жанров/тегов из Steam API
    и считает cosine similarity для всех игр.
    """
    if rated_df.empty or unrated_df.empty:
        return pd.DataFrame()
    
    # Собираем текстовое описание для каждой игры
    all_games = {}
    for _, row in rated_df.iterrows():
        appid = row['game_appid']
        genres = row.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(',') if g.strip()]
        elif not isinstance(genres, list):
            genres = []
        text = ' '.join(genres)
        all_games[appid] = text
    
    for _, row in unrated_df.iterrows():
        appid = row['game_appid']
        if appid not in all_games:
            genres = row.get('genres', [])
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(',') if g.strip()]
            elif not isinstance(genres, list):
                genres = []
            text = ' '.join(genres)
            all_games[appid] = text
    
    if not all_games:
        return pd.DataFrame()
    
    # Строим TF-IDF матрицу
    appids = list(all_games.keys())
    texts = [all_games[aid] for aid in appids]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Строим профиль пользователя как средневзвешенный вектор его любимых игр
    rated_appids = rated_df['game_appid'].tolist()
    ratings = rated_df['rating'].tolist() if 'rating' in rated_df.columns else [1.0] * len(rated_appids)
    
    # Индексы оценённых игр в матрице
    appid_to_idx = {aid: i for i, aid in enumerate(appids)}
    rated_indices = [appid_to_idx[aid] for aid in rated_appids if aid in appid_to_idx]
    valid_ratings = [ratings[i] for i, aid in enumerate(rated_appids) if aid in appid_to_idx]
    
    if not rated_indices:
        return pd.DataFrame()
    
    # Средневзвешенный профиль
    rated_vectors = tfidf_matrix[rated_indices].toarray()
    weights = np.array(valid_ratings, dtype=float)
    user_profile = np.average(rated_vectors, axis=0, weights=weights)
    
    # Считаем косинусное сходство со всеми играми
    similarities = cosine_similarity(user_profile.reshape(1, -1), tfidf_matrix).flatten()
    
    # Исключаем уже оценённые игры
    rated_set = set(rated_appids)
    results = []
    for i, appid in enumerate(appids):
        if appid not in rated_set:
            results.append({
                'game_appid': appid,
                'similarity_score': float(similarities[i]),
                'prediction': 3.5,  # Средний рейтинг как fallback
                'weighted_predition': float(similarities[i]) * 3.5,
            })
    
    result_df = pd.DataFrame(results)
    
    # Добавляем названия игр
    name_map = {}
    for _, row in pd.concat([rated_df, unrated_df]).iterrows():
        name_map[row['game_appid']] = row.get('game_name', 'Unknown')
    result_df['game_name'] = result_df['game_appid'].map(name_map).fillna('Unknown')
    
    return result_df.sort_values('weighted_predition', ascending=False)


# ==============================================================================
# 4. ЗАГРУЗКА ИЗОБРАЖЕНИЙ
# ==============================================================================
@st.cache_data(ttl=3600)
def load_game_image(app_id, img_url=None):
    """Загружает изображение игры в высоком качестве."""
    try:
        steam_cdn_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/library_600x900_2x.jpg"
        response = requests.get(steam_cdn_url, timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        
        steam_cdn_fallback = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"
        response = requests.get(steam_cdn_fallback, timeout=5)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
    except:
        pass
    
    if img_url and pd.notna(img_url) and str(img_url).strip() != "":
        try:
            response = requests.get(img_url, timeout=5)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except:
            pass
    
    return Image.new('RGB', (600, 900), color=(27, 40, 56))


# ==============================================================================
# 5. ФУНКЦИИ ЗАГРУЗКИ ДАННЫХ И МОДЕЛЕЙ
# ==============================================================================
@st.cache_data
def load_latest_parquet_pandas(subfolder):
    """Умная загрузка parquet с поддержкой вложенной структуры Spark."""
    folder_path = os.path.join(base_db_dir, subfolder)
    
    if not os.path.exists(folder_path):
        return pd.DataFrame()
        
    items = glob.glob(os.path.join(folder_path, '*'))
    if not items:
        return pd.DataFrame()
    
    latest_item = max(items, key=os.path.getctime)
    
    if os.path.isfile(latest_item) and latest_item.endswith('.parquet'):
        return pd.read_parquet(latest_item)
        
    if os.path.isdir(latest_item):
        parquet_files = glob.glob(os.path.join(latest_item, '*.parquet'))
        if not parquet_files:
            try:
                return pd.read_parquet(latest_item)
            except:
                return pd.DataFrame()
        
        dfs = []
        for f in parquet_files:
            try:
                dfs.append(pd.read_parquet(f))
            except:
                pass
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
        
    return pd.DataFrame()


@st.cache_data
def load_unrated_for_user(steam_id):
    """Ленивая загрузка unrated: читает только нужного пользователя."""
    folder_path = os.path.join(base_db_dir, 'unrated')
    
    if not os.path.exists(folder_path):
        return pd.DataFrame()
        
    items = glob.glob(os.path.join(folder_path, '*'))
    if not items:
        return pd.DataFrame()
        
    latest_item = max(items, key=os.path.getctime)
    
    if os.path.isdir(latest_item):
        parquet_files = glob.glob(os.path.join(latest_item, '*.parquet'))
    else:
        parquet_files = [latest_item]
        
    user_unrated = []
    for f in parquet_files:
        try:
            batch = pd.read_parquet(f)
            filtered = batch[batch['user_steamid'] == steam_id]
            if not filtered.empty:
                user_unrated.append(filtered)
        except:
            pass
            
    if user_unrated:
        return pd.concat(user_unrated, ignore_index=True)
    return pd.DataFrame()


@st.cache_resource
def load_local_models():
    """Загрузка обученных моделей."""
    hft_path = os.path.join(base_model_dir, 'hft_hybrid_model.pkl')
    tfidf_path = os.path.join(base_model_dir, 'tfidf_model.pkl')
    matrix_path = os.path.join(base_model_dir, 'tfidf_matrix.pkl')
    unique_path = os.path.join(base_model_dir, 'df_unique.pkl')
    
    missing_files = []
    for path in [hft_path, tfidf_path, matrix_path, unique_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        return None, None, None, None
    
    with open(hft_path, 'rb') as f:
        collab_model = pickle.load(f)
    with open(tfidf_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(matrix_path, 'rb') as f:
        matrix = pickle.load(f)
    with open(unique_path, 'rb') as f:
        uniq_tfidf = pickle.load(f)
        
    return collab_model, vectorizer, matrix, uniq_tfidf


# ==============================================================================
# 6. ЗАГРУЗКА БАЗОВЫХ ДАТАСЕТОВ
# ==============================================================================
users_df = load_latest_parquet_pandas('users')
games_df = load_latest_parquet_pandas('games')
games_df_clean = games_df.drop_duplicates(subset=['game_appid'])

df_rated_all = load_latest_parquet_pandas('rated')

if 'user_personaname' in users_df.columns:
    users_df['user_personaname'] = users_df['user_personaname'].apply(
        lambda x: x.encode('utf-8').decode('utf-8') if isinstance(x, str) else str(x)
    )

if not df_rated_all.empty and 'user_steamid' in df_rated_all.columns:
    valid_steam_ids = set(df_rated_all['user_steamid'].unique())
else:
    valid_steam_ids = set()

if not users_df.empty and 'user_steamid' in users_df.columns:
    active_users_df = users_df[users_df['user_steamid'].isin(valid_steam_ids)].copy()
else:
    active_users_df = pd.DataFrame()

if active_users_df.empty and not users_df.empty:
    active_users_df = users_df.copy()


# ==============================================================================
# 7. БОКОВОЙ ИНТЕРФЕЙС
# ==============================================================================
with st.sidebar:
    st.header("🕹️ Управление пользователями")
    
    mode = st.radio(
        "Режим работы:",
        ["Выбрать из базы данных", "🔍 Новый пользователь (Steam API)"]
    )
    
    st.divider()
    
    user_id_input = None
    user_name_input = "Новый пользователь"
    user_avatar_url = None
    user_profile_url = None
    user_country = None
    unrated_df = pd.DataFrame()
    rated_df = pd.DataFrame()
    recent_games_data = None
    games_details_list = None
    is_cold_start = False

    if mode == "Выбрать из базы данных":
        if active_users_df.empty:
            st.warning("⚠️ База пользователей пуста!")
        else:
            user_options = []
            for _, user_row in active_users_df.iterrows():
                steam_id = user_row['user_steamid']
                persona_name = str(user_row.get('user_personaname', 'Unknown'))
                persona_name = persona_name.replace('\x00', '').strip()
                if not persona_name or persona_name == 'nan':
                    persona_name = f"User_{steam_id[-6:]}"
                user_options.append(f"{persona_name} ({steam_id})")
            
            selected_option = st.selectbox('Выберите пользователя', user_options, index=0)
            
            user_id_input = selected_option.split('(')[-1].rstrip(')')
            
            user_name_match = active_users_df[active_users_df['user_steamid'] == user_id_input]['user_personaname'].values
            user_name_input = str(user_name_match[0]) if len(user_name_match) > 0 else "Unknown User"
            user_name_input = user_name_input.replace('\x00', '').strip()
            
            unrated_df = load_unrated_for_user(user_id_input)
            rated_df = df_rated_all[df_rated_all['user_steamid'] == user_id_input].copy()

    else:
        st.info("💡 Введите SteamID64, vanity URL или просто имя профиля")
        st.caption("Примеры:")
        st.caption("• `76561198999999999` (SteamID64)")
        st.caption("• `steamcommunity.com/id/sukadedins1241`")
        st.caption("• `sukadedins1241` (просто имя)")
        
        steam_input = st.text_input(
            "SteamID / URL / Имя профиля",
            placeholder="sukadedins1241 или 76561198...",
            help="Поддерживается SteamID64, vanity URL или просто имя профиля"
        )
        
        fetch_button = st.button("🔍 Получить данные из Steam", type="primary", use_container_width=True)
        
        if fetch_button and steam_input.strip():
            # Парсим ввод и получаем SteamID64
            with st.spinner("🔎 Преобразование vanity URL в SteamID64..."):
                steam_id_64 = parse_steam_input(steam_input)
            
            if not steam_id_64:
                st.error("❌ Не удалось определить SteamID64. Проверьте правильность ввода.")
            else:
                user_id_input = steam_id_64
                is_cold_start = True
                
                st.info(f"✅ SteamID64: `{steam_id_64}`")
                
                with st.spinner("🌐 Запрос к Steam API..."):
                    steam_data = get_steam_user_data(steam_id_64)
                
                if steam_data['error']:
                    st.error(steam_data['error'])
                else:
                    # Профиль
                    profile = steam_data['profile']
                    if profile:
                        user_name_input = profile.get('personaname', 'Unknown')
                        user_avatar_url = profile.get('avatar', '')
                        user_profile_url = profile.get('profileurl', '')
                        user_country = profile.get('loccountrycode', '')
                    
                    # Игры за 2 недели
                    recent_games_data = steam_data.get('recent_games')
                    
                    # Все игры с деталями
                    owned = steam_data.get('owned_games')
                    if owned:
                        unrated_df = owned['unrated_df']
                        rated_df = owned['rated_df']
                        games_details_list = owned['games_details']
                        
                        st.success(
                            f"✅ Получено данных:\n"
                            f"   • Профиль: **{user_name_input}**\n"
                            f"   • Всего игр: **{owned['game_count']}**\n"
                            f"   • Сыграно (>2ч): **{len(rated_df)}**\n"
                            f"   • Не сыграно: **{len(unrated_df)}**"
                        )
                        
                        if recent_games_data and recent_games_data['games']:
                            st.info(
                                f"🎮 Играл за последние 2 недели: **{len(recent_games_data['games'])}** игр"
                            )
                    else:
                        st.error("❌ Не удалось получить библиотеку игр")


# ==============================================================================
# 8. ГЛАВНЫЙ ЭКРАН
# ==============================================================================
st.title('🎮 Steam Intelligent Recommender System')
st.caption("Программный комплекс гибридной рекомендательной системы компьютерных игр в рамках ВКР")
st.divider()

# Отображение профиля пользователя (для Cold Start)
if is_cold_start and user_id_input:
    profile_col1, profile_col2 = st.columns([1, 5])
    
    with profile_col1:
        if user_avatar_url:
            try:
                avatar_response = requests.get(user_avatar_url, timeout=5)
                if avatar_response.status_code == 200:
                    avatar_img = Image.open(BytesIO(avatar_response.content))
                    st.image(avatar_img, width=128)
            except:
                pass
    
    with profile_col2:
        st.markdown(f"### 👤 {user_name_input}")
        info_parts = []
        if user_country:
            info_parts.append(f"🌍 {user_country}")
        if user_profile_url:
            info_parts.append(f"[🔗 Профиль в Steam]({user_profile_url})")
        if info_parts:
            st.markdown(" | ".join(info_parts))
    
    # Блок "Играл за последние 2 недели"
    if recent_games_data and recent_games_data['games']:
        st.markdown("#### 🎮 Играл за последние 2 недели")
        
        recent_cols = st.columns(min(5, len(recent_games_data['games'])))
        for idx, game in enumerate(recent_games_data['games'][:5]):
            with recent_cols[idx]:
                icon_url = f"https://media.steamstatic.com/steamcommunity/public/images/apps/{game['appid']}/{game['img_icon_url']}.jpg"
                try:
                    icon_response = requests.get(icon_url, timeout=3)
                    if icon_response.status_code == 200:
                        icon_img = Image.open(BytesIO(icon_response.content))
                        st.image(icon_img, width=64)
                except:
                    pass
                
                st.caption(f"**{game['name']}**")
                hours = game['playtime_2weeks'] / 60
                if hours < 1:
                    st.caption(f"⏱️ {game['playtime_2weeks']} мин")
                else:
                    st.caption(f"⏱️ {hours:.1f} ч")
        
        st.divider()

# Основная логика рекомендаций
if user_id_input and (not unrated_df.empty or not rated_df.empty):
    with st.spinner('🧠 Интеллектуальный движок рассчитывает персональный ТОП-10...'):
        
        # ======================================================================
        # РЕЖИМ COLD START: используем контентную модель на лету
        # ======================================================================
        if is_cold_start and games_details_list is not None:
            st.info("🆕 Режим Cold Start: строим профиль на основе жанров из Steam API")
            
            # Строим контентную модель на лету
            content_recs = build_content_model_for_new_user(rated_df, unrated_df, games_details_list)
            
            if content_recs.empty:
                st.warning("⚠️ Не удалось построить рекомендации на основе жанров")
                final_prediction = pd.DataFrame()
            else:
                # Добавляем изображения
                if not games_df_clean.empty and 'game_img_url' in games_df_clean.columns:
                    content_recs = pd.merge(
                        content_recs,
                        games_df_clean[['game_appid', 'game_img_url']],
                        on='game_appid',
                        how='left'
                    )
                
                final_prediction = content_recs.head(10)
            
            # Для аналитики
            final_prediction_all = content_recs
            real_top_results = rated_df.sort_values(by='game_playtime_forever', ascending=False) if 'game_playtime_forever' in rated_df.columns else rated_df
            
        # ======================================================================
        # РЕЖИМ ИЗ БАЗЫ: используем обученные модели
        # ======================================================================
        else:
            models = load_local_models()
            
            if models[0] is None:
                st.error("❌ Модели не загружены!")
                st.stop()
            
            collab_model, vectorizer, tfidf_matrix, df_unique_tfidf = models
            
            # Подготовка rated_df
            if 'game_name' in rated_df.columns:
                rated_df = rated_df.drop(columns=['game_name'])
            
            if not games_df_clean.empty and 'game_appid' in games_df_clean.columns and 'game_name' in games_df_clean.columns:
                rated_df = pd.merge(
                    rated_df, 
                    games_df_clean[['game_appid', 'game_name']], 
                    on='game_appid', 
                    how='left'
                )
            
            if 'game_name' not in rated_df.columns:
                rated_df['game_name'] = 'Unknown Game'
            else:
                rated_df['game_name'] = rated_df['game_name'].fillna('Unknown Game')
            
            # Маппинги моделей
            user_map = getattr(collab_model, 'user_map', getattr(collab_model, 'uid_map', None))
            item_map = getattr(collab_model, 'item_map', getattr(collab_model, 'iid_map', None))
            global_mean = getattr(collab_model, 'global_mean', 3.0)
            
            user_in_model = user_map is not None and user_id_input in user_map
            
            # Коллаборативные предсказания
            preds_unrated = []
            for row in unrated_df.itertuples():
                if user_in_model:
                    uidx = user_map.get(row.user_steamid)
                    iidx = item_map.get(row.game_appid) if item_map else None
                    score = float(collab_model.score(uidx, iidx)) if (uidx is not None and iidx is not None) else float(global_mean)
                else:
                    score = float(global_mean)
                preds_unrated.append(score)
            unrated_df['prediction'] = preds_unrated
            
            preds_rated = []
            for row in rated_df.itertuples():
                if user_in_model:
                    uidx = user_map.get(row.user_steamid)
                    iidx = item_map.get(row.game_appid) if item_map else None
                    score = float(collab_model.score(uidx, iidx)) if (uidx is not None and iidx is not None) else float(global_mean)
                else:
                    score = float(global_mean)
            preds_rated.append(score)
            rated_df['prediction'] = preds_rated
            
            # Контентный анализ
            u_top_games = rated_df[rated_df['rating'] == 5].copy() if 'rating' in rated_df.columns else pd.DataFrame()
            if u_top_games.empty and 'game_playtime_forever' in rated_df.columns:
                u_top_games = rated_df.sort_values(by='game_playtime_forever', ascending=False).head(3)
                
            cosine_weights = {}
            if not u_top_games.empty and tfidf_matrix is not None and df_unique_tfidf is not None:
                for favorite_game_id in u_top_games['game_appid']:
                    match = df_unique_tfidf[df_unique_tfidf['game_appid'] == favorite_game_id]
                    if match.empty:
                        continue
                    fav_idx = match.index[0]
                    sims = cosine_similarity(tfidf_matrix[fav_idx], tfidf_matrix).flatten()
                    for idx, game_row in df_unique_tfidf.iterrows():
                        g_id = game_row['game_appid']
                        if sims[idx] > cosine_weights.get(g_id, -1):
                            cosine_weights[g_id] = sims[idx]
                            
            # Расчет финальных весов
            unrated_df['similarity_score'] = unrated_df['game_appid'].map(cosine_weights).fillna(0.05)
            unrated_df['weighted_predition'] = unrated_df['prediction'] * unrated_df['similarity_score']
            
            # ФИЛЬТРАЦИЯ: Исключаем игры, которые уже есть у пользователя
            owned_games = set(rated_df['game_appid'].unique()) if not rated_df.empty else set()
            if not unrated_df.empty:
                unrated_df = unrated_df[~unrated_df['game_appid'].isin(owned_games)]
            
            if 'game_name' in unrated_df.columns:
                unrated_df = unrated_df.drop(columns=['game_name'])
            
            if not games_df_clean.empty and 'game_appid' in games_df_clean.columns:
                merge_cols = ['game_appid']
                if 'game_name' in games_df_clean.columns:
                    merge_cols.append('game_name')
                if 'game_img_url' in games_df_clean.columns:
                    merge_cols.append('game_img_url')
                
                unrated_df = pd.merge(
                    unrated_df, 
                    games_df_clean[merge_cols], 
                    on='game_appid', 
                    how='left'
                )
            
            if 'game_name' not in unrated_df.columns:
                unrated_df['game_name'] = 'Unknown Game'
            else:
                unrated_df['game_name'] = unrated_df['game_name'].fillna('Unknown Game')
            
            rated_df['similarity_score'] = rated_df['game_appid'].map(cosine_weights).fillna(0.05)
            rated_df['weighted_predition'] = rated_df['prediction'] * rated_df['similarity_score']

            final_prediction = unrated_df.sort_values(by='weighted_predition', ascending=False).head(10)
            
            if 'game_playtime_forever' in rated_df.columns:
                real_top_results = rated_df.sort_values(by='game_playtime_forever', ascending=False)
            else:
                real_top_results = rated_df.copy()
            
            final_prediction_all = pd.concat([unrated_df, rated_df], ignore_index=True)
            final_prediction_all = final_prediction_all.sort_values(
                by='weighted_predition', 
                ascending=False
            ).drop_duplicates(subset='game_appid')

        # ======================================================================
        # ОТРИСОВКА РЕЗУЛЬТАТОВ
        # ======================================================================
        if is_cold_start:
            st.warning("⚠️ Новый пользователь! Используем контентную модель на основе жанров из Steam API.")
            st.info("💡 Система получила данные через Steam API, извлекла жанры каждой игры и построила профиль на основе TF-IDF сходства.")

        st.markdown(f'## 🎮 Рекомендации для: **{user_name_input}**')
        if user_profile_url:
            st.caption(f"🔗 [Открыть профиль в Steam]({user_profile_url})")
        st.divider()

        if final_prediction.empty:
            st.info("ℹ️ Нет доступных игр для рекомендации.")
        else:
            for idx, row in enumerate(final_prediction.itertuples(), 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([1.5, 4, 2, 2])
                    
                    game_image = load_game_image(row.game_appid, getattr(row, 'game_img_url', None))
                    
                    with col1:
                        if game_image:
                            st.image(game_image, use_container_width=True)
                    
                    with col2:
                        st.markdown(f"### {idx}. **{row.game_name}**")
                        st.caption(f"AppID: {row.game_appid}")
                        
                        # Показываем жанры из Steam API если есть
                        if games_details_list:
                            for gd in games_details_list:
                                if gd['game_appid'] == row.game_appid and gd.get('genres'):
                                    st.caption(f"🎭 Жанры: {', '.join(gd['genres'][:5])}")
                                    break
                        elif not games_df_clean.empty and 'game_appid' in games_df_clean.columns:
                            game_info = games_df_clean[games_df_clean['game_appid'] == row.game_appid]
                            if not game_info.empty and 'game_genres' in game_info.columns:
                                genres = str(game_info.iloc[0]['game_genres'])[:100]
                                st.caption(f"🎭 Жанры: {genres}")
                    
                    with col3:
                        if is_cold_start:
                            st.markdown("**🎯 Similarity Score:**")
                            st.markdown(f"### {row.weighted_predition:.3f}")
                        else:
                            st.markdown("**🎯 Гибридный балл:**")
                            st.markdown(f"### {row.weighted_predition:.3f}")
                            st.caption(f"Коллаб: {row.prediction:.1f}")
                            st.caption(f"Контент: {row.similarity_score:.2f}")
                    
                    with col4:
                        st.write("")
                        st.write("")
                        st.markdown(
                            f'<a href="https://store.steampowered.com/app/{row.game_appid}" '
                            f'target="_blank" style="text-decoration: none;">'
                            f'<button style="background-color: #1a9fff; color: white; '
                            f'border: none; padding: 10px 20px; border-radius: 5px; '
                            f'cursor: pointer; width: 100%;">'
                            f'🏬 В магазин Steam</button></a>',
                            unsafe_allow_html=True
                        )
                    
                    st.markdown("---")

        # ======================================================================
        # АНАЛИТИЧЕСКИЙ КОНТУР
        # ======================================================================
        st.markdown('## 📊 Аналитический контур верификации')
        st.divider()

        if not final_prediction_all.empty and 'game_appid' in final_prediction_all.columns:
            predicted_top_10 = final_prediction_all['game_appid'].head(10).tolist()
            real_top_10 = real_top_results['game_appid'].head(10).tolist() if 'game_appid' in real_top_results.columns else []
            matches = set(predicted_top_10) & set(real_top_10)
            precision_at_10 = len(matches) / 10 if len(predicted_top_10) > 0 else 0
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="🎯 Precision@10", 
                    value=f"{precision_at_10 * 100:.0f}%",
                    help="Доля совпадений между предсказанным и реальным топом"
                )
            with col_m2:
                st.metric(label="📈 Игр в истории", value=f"{len(real_top_results)} шт.")
            with col_m3:
                st.metric(label="🕹️ Неоцененных игр", value=f"{len(unrated_df)} шт.")
            
            st.divider()
            col_an1, col_an2 = st.columns(2)
            
            with col_an1:
                st.write('**📋 История игр (Топ по времени):**')
                display_cols = ['game_appid', 'game_name']
                if 'game_playtime_forever' in real_top_results.columns:
                    display_cols.append('game_playtime_forever')
                
                display_df = real_top_results[display_cols].head(10)
                if 'game_playtime_forever' in display_df.columns:
                    display_df = display_df.rename(columns={'game_playtime_forever': 'Часы'})
                    display_df['Часы'] = display_df['Часы'].apply(lambda x: f"{x/60:.1f}ч")
                st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)
                
            with col_an2:
                st.write('**🔮 Топ рекомендаций:**')
                display_pred = final_prediction_all[['game_appid', 'game_name', 'weighted_predition']].head(10)
                display_pred = display_pred.rename(columns={'weighted_predition': 'Рейтинг'})
                display_pred['Рейтинг'] = display_pred['Рейтинг'].apply(lambda x: f"{x:.3f}")
                st.dataframe(display_pred.reset_index(drop=True), use_container_width=True, hide_index=True)

else:
    st.info('👈 Пожалуйста, выберите существующего пользователя или введите SteamID нового на боковой панели.')
    
    with st.expander("📖 Как пользоваться системой?"):
        st.markdown("""
        1. **Выберите пользователя** из базы данных в боковой панели
        2. Или **введите SteamID64 / vanity URL / имя профиля** для получения рекомендаций через Steam API
        3. Система автоматически получит данные о играх пользователя (включая игры за последние 2 недели)
        4. Изучите аналитический контур для верификации результатов
        """)
