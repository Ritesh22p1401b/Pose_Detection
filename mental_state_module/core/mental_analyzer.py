from mental_state_module.core.verdict_rules import classify


class MentalAnalyzer:
    def analyze(self, first_emotion, emotions):
        return classify(first_emotion, emotions)
