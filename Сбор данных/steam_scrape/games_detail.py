# This module scrapes game data from Steam and saves it to a JSON file.
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime 
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to save JSON data to a file with the current timestamp
def save_json(data, directory, prefix):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    with open(f'{directory}/{prefix}_{timestamp}.json', 'w') as file:
        json.dump(data, file, indent=4)

# Function to scrape game data
def scrape_game_data(game_id):
    # Construct the URL with the game ID
    url = f'https://store.steampowered.com/app/{game_id}'

    # Cookies and headers obtained with CURL from Steam using https://curlconverter.com/python/
    cookies = {
    'wants_mature_content': '1',
    'browserid': '341805644281391573',
    'timezoneOffset': '-10800,0',
    'steamLoginSecure': '76561198147702705%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MDAwQ18yNTY2QzJDNl8wMjAyNiIsICJzdWIiOiAiNzY1NjExOTgxNDc3MDI3MDUiLCAiYXVkIjogWyAid2ViOnN0b3JlIiBdLCAiZXhwIjogMTczMzExMjc5MywgIm5iZiI6IDE3MjQzODYxNjgsICJpYXQiOiAxNzMzMDI2MTY4LCAianRpIjogIjAwMDNfMjU2RkZDODdfMkVGMTEiLCAib2F0IjogMTczMjMwNDkyNywgInJ0X2V4cCI6IDE3NTAyODc2MDMsICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICIxNzcuMTQ1Ljk0LjM4IiwgImlwX2NvbmZpcm1lciI6ICIxNzcuMTQ1Ljk0LjM4IiB9.ZLXuj3fkff9a1PBuLDDyScWdCTLtNgcf2mZZ8uF4jgAEPyuGlPoO90o1dZLR1bEaJz_Gnc3VhmErG4xjRU1ZBw',
    'steamCountry': 'BR%7Cbcc8985ae058b1dd0644025303a9cdc9',
    'sessionid': '31e2582c175b29c59c8faccb',
    'bGameHighlightAutoplayDisabled': 'true',
    'timezoneName': 'America/Sao_Paulo',
    'Steam_Language': 'english',
    'recentapps': '%7B%222537590%22%3A1733028068%2C%222780980%22%3A1733026698%2C%2210%22%3A1733026216%2C%222694490%22%3A1733026211%7D',
    'birthtime': '767761201',
    'lastagecheckage': '1-May-1994',
    }

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'es-419,es;q=0.9,es-ES;q=0.8,en;q=0.7,en-GB;q=0.6,en-US;q=0.5,es-MX;q=0.4,pt-BR;q=0.3,pt;q=0.2',
        'Connection': 'keep-alive',
        # 'Cookie': 'wants_mature_content=1; browserid=341805644281391573; timezoneOffset=-10800,0; steamLoginSecure=76561198147702705%7C%7CeyAidHlwIjogIkpXVCIsICJhbGciOiAiRWREU0EiIH0.eyAiaXNzIjogInI6MDAwQ18yNTY2QzJDNl8wMjAyNiIsICJzdWIiOiAiNzY1NjExOTgxNDc3MDI3MDUiLCAiYXVkIjogWyAid2ViOnN0b3JlIiBdLCAiZXhwIjogMTczMzExMjc5MywgIm5iZiI6IDE3MjQzODYxNjgsICJpYXQiOiAxNzMzMDI2MTY4LCAianRpIjogIjAwMDNfMjU2RkZDODdfMkVGMTEiLCAib2F0IjogMTczMjMwNDkyNywgInJ0X2V4cCI6IDE3NTAyODc2MDMsICJwZXIiOiAwLCAiaXBfc3ViamVjdCI6ICIxNzcuMTQ1Ljk0LjM4IiwgImlwX2NvbmZpcm1lciI6ICIxNzcuMTQ1Ljk0LjM4IiB9.ZLXuj3fkff9a1PBuLDDyScWdCTLtNgcf2mZZ8uF4jgAEPyuGlPoO90o1dZLR1bEaJz_Gnc3VhmErG4xjRU1ZBw; steamCountry=BR%7Cbcc8985ae058b1dd0644025303a9cdc9; sessionid=31e2582c175b29c59c8faccb; bGameHighlightAutoplayDisabled=true; timezoneName=America/Sao_Paulo; Steam_Language=english; recentapps=%7B%222537590%22%3A1733028068%2C%222780980%22%3A1733026698%2C%2210%22%3A1733026216%2C%222694490%22%3A1733026211%7D; birthtime=767761201; lastagecheckage=1-May-1994',
        'Referer': 'https://store.steampowered.com/agecheck/app/4720/',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0',
        'sec-ch-ua': '"Microsoft Edge";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    params = {
        'snr': '1_direct-navigation__',
    }

    response = requests.get(url, cookies=cookies, headers=headers, params=params, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract release date
    rightcol = soup.find('div', class_='rightcol')
    release_date_raw = rightcol.find('div', class_='release_date')\
                               .find('div', class_='date')\
                               .text
    release_date = datetime.strptime(release_date_raw, '%d %b, %Y').strftime('%Y-%m-%d')

    # Extract developer and publisher
    dev_rows = soup.find_all('div', class_='dev_row')
    result = {}
    for row in dev_rows:
        subtitle_div = row.find('div', class_='subtitle column')
        if subtitle_div:
            subtitle = subtitle_div.get_text(strip=True).rstrip(':')
            summary_div = row.find('div', class_='summary column')
            value = summary_div.get_text(strip=True) if summary_div else None
            result[subtitle] = value

    developer = result.get('Developer')
    publisher = result.get('Publisher')

    #Extract genres
    genres_div = soup.find('div', class_='details_block')
    genres_span = genres_div.find('span', attrs={'data-panel': '{"flow-children":"row"}'})
    genres = [a.get_text(strip=True) for a in genres_span.find_all('a')]

    # Extract tags
    tags_div = rightcol.find('div', class_='glance_tags popular_tags')
    tags = [a.get_text(strip=True) for a in tags_div.find_all('a', class_='app_tag')]

    # Extract rating
    rating_div = rightcol.find('div', class_='user_reviews')
    rating = None
    if rating_div:
        rating_span = rating_div.find('div', class_='summary column').find('span')
        if rating_span:
            rating = rating_span.get_text(strip=True)

    return {
        'gameid': game_id,
        'release_date': release_date,
        'developer': developer,
        'publisher': publisher,
        'genres': genres,
        'tags': tags,
        'rating': rating
    }

# Function to scrape games in parallel to reduce scraping time
def scrape_games_parallel(game_ids, max_workers=10):
    results = []
    failed_games = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_game_id = {executor.submit(scrape_game_data, game_id): game_id for game_id in game_ids}
        for future in tqdm(as_completed(future_to_game_id), total=len(game_ids), desc='Scraping Games'):
            game_id = future_to_game_id[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                print(f'Error scraping game ID {game_id}: {e}')
                failed_games.append(game_id)

    return results, failed_games

# Function to scrape games in batches to avoid memory issues
def scrape_games_in_batches(game_ids, batch_size=5000, max_workers=1200):
    all_results = []
    all_failed_games = []

    for i in range(0, len(game_ids), batch_size):
        batch = game_ids[i:i + batch_size]
        print(f'Processing batch {i // batch_size + 1}/{-(-len(game_ids) // batch_size)}...')

        results, failed_games = scrape_games_parallel(batch, max_workers=max_workers)
        all_results.extend(results)
        all_failed_games.extend(failed_games)

        # Save batch results to avoid losing data if the script is interrupted
        save_json(results, 'db/games_detail', f'games_detail_batch_{i // batch_size + 1}')
        save_json(failed_games, 'db/failed_games', f'failed_games_batch_{i // batch_size + 1}')

    return all_results, all_failed_games