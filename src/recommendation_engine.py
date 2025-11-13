class RecommendationEngine:
    def apply_business_rules(self) -> FilteredRecommendations # Применение бизнес-правил
    def diversify_recommendations(self) -> DiverseResults     # Диверсификация выдачи
    def calculate_explanation_scores(self) -> Explanation     # Объяснение рекомендаций
