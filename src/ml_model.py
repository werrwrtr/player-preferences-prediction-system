# Улучшенная ML модель для рекомендаций
class ImprovedRecommendationModel:
    def __init__(self):
        self.user_profiles = {}
    
    def create_user_profile(self, player_data, game_history):
        """
        Создает профиль пользователя на основе данных
        """
        try:
            player_id = player_data.get('player_id')
            if not player_id:
                return None
            
            # Анализируем игровые предпочтения
            total_play_time = sum([sess.get('duration', 0) for sess in game_history])
            favorite_genres = self._analyze_genres(game_history)
            
            profile = {
                'player_id': player_id,
                'total_play_time': total_play_time,
                'favorite_genres': favorite_genres,
                'skill_level': self._determine_skill_level(total_play_time),
                'preferred_game_types': self._analyze_game_types(game_history)
            }
            
            self.user_profiles[player_id] = profile
            return profile
            
        except Exception as e:
            print(f"Ошибка создания профиля: {e}")
            return None
    
    def _analyze_genres(self, game_history):
        """Анализирует любимые жанры"""
        genre_count = {}
        for session in game_history:
            genre = session.get('genre', 'unknown')
            genre_count[genre] = genre_count.get(genre, 0) + 1
        
        # Возвращаем топ-3 жанра
        return sorted(genre_count.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _determine_skill_level(self, total_play_time):
        """Определяет уровень навыка игрока"""
        if total_play_time > 1000:
            return 'expert'
        elif total_play_time > 500:
            return 'advanced'
        elif total_play_time > 100:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _analyze_game_types(self, game_history):
        """Анализирует предпочтения по типам игр"""
        types = {'singleplayer': 0, 'multiplayer': 0, 'coop': 0}
        for session in game_history:
            game_type = session.get('game_type', 'singleplayer')
            if game_type in types:
                types[game_type] += 1
        
        return max(types.items(), key=lambda x: x[1])[0]
    
    def generate_recommendations(self, player_id):
        """
        Генерирует персональные рекомендации
        """
        if player_id not in self.user_profiles:
            return ["Популярные новинки", "Бестселлеры"]
        
        profile = self.user_profiles[player_id]
        recommendations = []
        
        # Рекомендации на основе жанров
        for genre, _ in profile['favorite_genres']:
            if genre == 'action':
                recommendations.append("Новые экшн-игры")
            elif genre == 'rpg':
                recommendations.append("РПГ с глубоким сюжетом")
            elif genre == 'strategy':
                recommendations.append("Стратегии с тактическими боями")
        
        # Рекомендации на основе уровня навыка
        if profile['skill_level'] in ['advanced', 'expert']:
            recommendations.append("Сложные игры для опытных игроков")
        else:
            recommendations.append("Игры для начинающих")
        
        # Рекомендации на основе типа игры
        if profile['preferred_game_types'] == 'multiplayer':
            recommendations.append("Мультиплеерные хиты")
        elif profile['preferred_game_types'] == 'coop':
            recommendations.append("Кооперативные приключения")
        
        return recommendations[:5]  # Возвращаем топ-5 рекомендаций

# Тестирование
if __name__ == "__main__":
    model = ImprovedRecommendationModel()
    
    # Тестовые данные
    player_data = {'player_id': 'user123'}
    game_history = [
        {'genre': 'action', 'duration': 50, 'game_type': 'multiplayer'},
        {'genre': 'rpg', 'duration': 120, 'game_type': 'singleplayer'},
        {'genre': 'action', 'duration': 30, 'game_type': 'multiplayer'},
        {'genre': 'strategy', 'duration': 80, 'game_type': 'singleplayer'}
    ]
    
    profile = model.create_user_profile(player_data, game_history)
    print("Профиль игрока:", profile)
    
    recommendations = model.generate_recommendations('user123')
    print("Рекомендации:", recommendations)