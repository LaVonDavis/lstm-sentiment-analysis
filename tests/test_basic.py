import unittest
from src.data_loading import load_imdb_data
from src.model import build_lstm_model

class TestModel(unittest.TestCase):
    def test_data_loading(self):
        (x_train, _), (x_test, _) = load_imdb_data()
        self.assertEqual(x_train.shape[0], 25000)
        self.assertEqual(x_test.shape[0], 25000)

    def test_model_creation(self):
        model = build_lstm_model()
        self.assertIsInstance(model, Model)
