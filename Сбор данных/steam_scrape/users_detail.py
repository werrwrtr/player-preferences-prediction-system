# This module fetches user data from Steam API
import requests

# Function to fetch user data using Steam API
def get_users_detail(**kwargs):
    url = 'http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002'
    response = requests.get(url,params=kwargs)
    return response 