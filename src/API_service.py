"""
Простой API сервис для системы рекомендаций
"""
from flask import Flask, jsonify, request
import json
import os

app = Flask(__name__)

# Простое хранилище данных в файле
DATA_FILE = 'player_data.json'

def load_data():
    """Загружает данные из файла"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'players': {}, 'recommendations': {}}

def save_data(data):
    """Сохраняет данные в файл"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route('/')
def home():
    """Главная страница API"""
    return jsonify({
        'service': 'Player Recommendations API',
        'version': '1.0',
        'endpoints': {
            '/players': 'GET - список игроков',
            '/players': 'POST - добавить игрока', 
            '/players/<id>/recommendations': 'GET - рекомендации для игрока'
        }
    })

@app.route('/players', methods=['GET'])
def get_players():
    """Возвращает список всех игроков"""
    data = load_data()
    return jsonify({
        'players': list(data['players'].keys()),
        'total': len(data['players'])
    })

@app.route('/players', methods=['POST'])
def add_player():
    """Добавляет нового игрока"""
    try:
        player_data = request.get_json()
        
        if not player_data or 'player_id' not in player_data:
            return jsonify({'error': 'Необходим player_id'}), 400
        
        data = load_data()
        player_id = player_data['player_id']
        
        # Сохраняем данные игрока
        data['players'][player_id] = player_data
        
        # Генерируем простые рекомендации
        from ml_model import ImprovedRecommendationModel
        model = ImprovedRecommendationModel()
        
        # Фиктивная история игр для демонстрации
        sample_history = [
            {'genre': 'action', 'duration': 60, 'game_type': 'multiplayer'},
            {'genre': 'rpg', 'duration': 90, 'game_type': 'singleplayer'}
        ]
        
        model.create_user_profile(player_data, sample_history)
        recommendations = model.generate_recommendations(player_id)
        
        data['recommendations'][player_id] = recommendations
        save_data(data)
        
        return jsonify({
            'status': 'success',
            'player_id': player_id,
            'recommendations': recommendations
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/players/<player_id>/recommendations', methods=['GET'])
def get_recommendations(player_id):
    """Получает рекомендации для игрока"""
    data = load_data()
    
    if player_id not in data['recommendations']:
        return jsonify({'error': 'Игрок не найден'}), 404
    
    return jsonify({
        'player_id': player_id,
        'recommendations': data['recommendations'][player_id]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({'status': 'healthy', 'service': 'recommendations'})

if __name__ == '__main__':
    print("=== Запуск API сервиса рекомендаций ===")
    print("Доступные endpoints:")
    print("  GET  / - информация о API")
    print("  GET  /players - список игроков") 
    print("  POST /players - добавить игрока")
    print("  GET  /players/<id>/recommendations - получить рекомендации")
    print("  GET  /health - проверка здоровья")
    print("\nСервер запущен: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)