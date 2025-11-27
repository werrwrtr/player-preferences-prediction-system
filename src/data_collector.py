# Добавляем в начало файла (или создаем если его нет)
def safe_data_collection(player_data):
    """
    Безопасный сбор данных с обработкой ошибок
    """
    try:
        # Проверяем входные данные
        if not player_data or 'player_id' not in player_data:
            print("Ошибка: некорректные данные игрока")
            return None
        
        # Имитируем сбор данных
        print(f"Собираем данные для игрока: {player_data['player_id']}")
        
        # Возвращаем фиктивные данные
        return {
            'player_id': player_data['player_id'],
            'data_quality': 'high',
            'timestamp': '2025-01-01 12:00:00'
        }
    
    except Exception as e:
        print(f"Критическая ошибка при сборе данных: {e}")
        return None

# Добавляем в конец файла для тестирования
if __name__ == "__main__":
    # Тест правильных данных
    good_data = {'player_id': 'test123'}
    result = safe_data_collection(good_data)
    print("Результат с хорошими данными:", result)
    
    # Тест плохих данных
    bad_data = None
    result2 = safe_data_collection(bad_data)
    print("Результат с плохими данными:", result2)