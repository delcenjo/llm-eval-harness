from llmeval.predictors import KeywordBaseline


def test_keyword_baseline_routes_clear_cases():
    baseline = KeywordBaseline()
    assert baseline.predict("I want a refund for my invoice") == "billing"
    assert baseline.predict("the app keeps crashing with an error") == "technical"
    assert baseline.predict("I forgot my password") == "account"
    assert baseline.predict("please cancel my subscription") == "cancellation"
    assert baseline.predict("hello, how are you today?") == "other"
