# This module fetches user data from Steam API
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function to fetch friends data using Steam API
def get_friends_data(api_key, steamid):
    url = 'http://api.steampowered.com/ISteamUser/GetFriendList/v0001/'
    try:
        response = requests.get(url, params={'key': api_key, 'steamid': steamid, 'relationship': 'all'}, timeout=5)
        response.raise_for_status()
        data = response.json()
        return steamid, data.get('friendslist', {}).get('friends', [])
    except Exception as e:
        print(f'Error fetching data for {steamid}: {e}')
        return steamid, []

# Function to work in parallel to fetch friends data
def fetch_friend_data(api_key, initial_steamid, max_unique_ids, max_workers=10):
    unique_ids = set()
    friends_list = []
    steamid_queue = [initial_steamid]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=max_unique_ids) as pbar:
            while steamid_queue and len(unique_ids) < max_unique_ids:
                batch = []
                while steamid_queue and len(batch) < max_workers:
                    steamid = steamid_queue.pop(0)
                    if steamid not in unique_ids:
                        batch.append(steamid)

                # Submit tasks
                futures = [executor.submit(get_friends_data, api_key, sid) for sid in batch]

                for future in as_completed(futures):
                    steamid, friends = future.result()
                    if steamid in unique_ids:
                        continue

                    unique_ids.add(steamid)
                    pbar.update(1)

                    friends_list.append({
                        'parent_steamid': steamid,
                        'friendslist': {'friends': friends}
                    })

                    for friend in friends:
                        if friend['steamid'] not in unique_ids and len(unique_ids) + len(steamid_queue) < max_unique_ids:
                            steamid_queue.append(friend['steamid'])

    return {'friends_list': friends_list}, list(unique_ids) 
