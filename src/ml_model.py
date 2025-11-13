class MLModel:
    def train_collaborative_model(self) -> ModelMetrics       # Обучение коллаборативной фильтрации
    def predict_preferences(self) -> List[Prediction]         # Прогнозирование предпочтений
    def evaluate_model_performance(self) -> PerformanceReport # Оценка качества модели
    def retrain_on_new_data(self) -> bool                     # Дообучение на новых данных
