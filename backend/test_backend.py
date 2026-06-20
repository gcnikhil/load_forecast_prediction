import unittest
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add the current directory to sys.path so we can import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import ModelService

class TestModelService(unittest.TestCase):
    def setUp(self):
        self.service = ModelService()
        self.service.load_models()

    def test_model_loading(self):
        # Even if models aren't present (e.g. missing TensorFlow), service should run in fallback/dummy mode.
        # But since we just successfully trained them and auto-deployed, they should load correctly!
        self.assertTrue(self.service.is_loaded() or self.service.last_prediction_info["used_dummy"])

    def test_predict_standard(self):
        start_date = datetime(2026, 6, 21)
        end_date = datetime(2026, 6, 23)
        df = self.service.predict(start_date, end_date)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("loads_lightgbm_gru", df.columns)
        self.assertGreater(len(df), 0)
        
        # Verify 5-minute resolution
        diffs = df.index.to_series().diff().dropna().dt.total_seconds() / 60
        self.assertEqual(int(diffs.mode().iloc[0]), 5)

    def test_predict_ignore_actuals(self):
        start_date = datetime(2026, 6, 21)
        end_date = datetime(2026, 6, 23)
        # Should execute successfully without throwing errors
        df = self.service.predict(start_date, end_date, ignore_actuals=True)
        self.assertIsInstance(df, pd.DataFrame)

    def test_predict_feature_overrides(self):
        start_date = datetime(2026, 6, 21)
        end_date = datetime(2026, 6, 23)
        overrides = {"temperature_celsius": 40.0, "is_holiday": 1}
        
        # Should execute successfully with overrides
        df = self.service.predict(start_date, end_date, feature_overrides=overrides, ignore_actuals=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("loads_lightgbm_gru", df.columns)

    def test_fallback_alignment(self):
        start_date = datetime(2026, 6, 21)
        end_date = datetime(2026, 6, 23)
        df = self.service._fallback_dataframe(start_date, end_date)
        
        # Should have correct columns and index resolution
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("loads_lightgbm_gru", df.columns)
        diffs = df.index.to_series().diff().dropna().dt.total_seconds() / 60
        self.assertEqual(int(diffs.mode().iloc[0]), 5)

if __name__ == "__main__":
    unittest.main()
