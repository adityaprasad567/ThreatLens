from src.prediction import load_model, predict_url
from src.train_model import registrable_domain


def test_domain_grouping_key():
    assert registrable_domain('https://a.example.co.uk/login') == 'example.co.uk'
    assert registrable_domain('https://sub.example.com/path') == 'example.com'


def test_saved_model_uses_full_url_features():
    bundle = load_model()
    assert bundle['training_mode'] == 'full_url_domain_grouped'
    assert bundle['uses_host_text'] is True
    assert bundle['text_input'] == 'complete_normalized_url'


def test_legitimate_url_prediction_runs():
    result = predict_url('https://www.google.com')
    assert result['ok'] is True
    assert result['prediction'] in {'Legitimate', 'Phishing'}


def test_synthetic_brand_impersonation_is_not_reassured():
    result = predict_url('http://amazon-prize-claim.example.com')
    assert result['ok'] is True
    assert result['site_verdict']['label'] != 'Likely REAL'
