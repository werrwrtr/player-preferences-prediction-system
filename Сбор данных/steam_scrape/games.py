# This module fetches game data from Steam API
import requests

# Function to fetch game data from users using Steam API
def get_games_data(**kwargs):
    url = 'https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/'
    response = requests.get(url,params=kwargs)
    return response 