"""
test_pipeline.py

Unit tests for:
    - src/data_ingestion.py
    - src/data_preprocessing.py

Run with:
    pytest tests/test_pipeline.py -v
"""

import os
import sys
import tempfile
import shutil
import unittest

import pandas as pd
import numpy as np
import yaml

# ------------------------------------------------------------------ #
# Make sure `src` is importable regardless of where pytest is invoked
# from. The pipeline code lives in /src (relative to the repo root).
# ------------------------------------------------------------------ #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src import data_ingestion  # noqa: E402
from src import data_preprocessing  # noqa: E402


class TestDataIngestion(unittest.TestCase):
    """Unit tests for src/data_ingestion.py"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        self.sample_df = pd.DataFrame(
            {
                "price": [100000, 200000, 300000, 400000, 500000, 600000],
                "area": [1000, 1500, 2000, 2500, 3000, 3500],
                "mainroad": ["yes", "no", "yes", "no", "yes", "no"],
            }
        )

        self.csv_path = os.path.join(self.tmp_dir, "Housing.csv")
        self.sample_df.to_csv(self.csv_path, index=False)

        self.params_path = os.path.join(self.tmp_dir, "params.yaml")
        self.params = {
            "data_ingestion": {
                "raw_data_path": self.csv_path,
                "raw_dir": os.path.join(self.tmp_dir, "raw"),
                "test_size": 0.2,
                "random_state": 42,
            }
        }
        with open(self.params_path, "w") as f:
            yaml.dump(self.params, f)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ---------------------- load_params ---------------------- #
    def test_load_params_success(self):
        params = data_ingestion.load_params(self.params_path)
        self.assertIn("data_ingestion", params)
        self.assertEqual(params["data_ingestion"]["test_size"], 0.2)

    def test_load_params_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            data_ingestion.load_params(os.path.join(self.tmp_dir, "missing.yaml"))

    def test_load_params_invalid_yaml(self):
        bad_path = os.path.join(self.tmp_dir, "bad.yaml")
        with open(bad_path, "w") as f:
            f.write("key: [unbalanced brackets\n")
        with self.assertRaises(yaml.YAMLError):
            data_ingestion.load_params(bad_path)

    # ----------------------- load_data ------------------------ #
    def test_load_data_success(self):
        df = data_ingestion.load_data(self.csv_path)
        self.assertEqual(df.shape, self.sample_df.shape)
        self.assertListEqual(list(df.columns), list(self.sample_df.columns))

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            data_ingestion.load_data(os.path.join(self.tmp_dir, "missing.csv"))

    # ---------------------- validate_data ---------------------- #
    def test_validate_data_passes_on_valid_df(self):
        # Should not raise
        data_ingestion.validate_data(self.sample_df)

    def test_validate_data_raises_on_empty_df(self):
        with self.assertRaises(ValueError):
            data_ingestion.validate_data(pd.DataFrame())

    def test_validate_data_raises_on_missing_target_column(self):
        df = self.sample_df.drop(columns=["price"])
        with self.assertRaises(ValueError):
            data_ingestion.validate_data(df)

    # ------------------------ save_data ------------------------ #
    def test_save_data_creates_train_and_test_files(self):
        out_dir = os.path.join(self.tmp_dir, "raw_out")
        train_df = self.sample_df.iloc[:4]
        test_df = self.sample_df.iloc[4:]

        data_ingestion.save_data(train_df, test_df, out_dir)

        train_path = os.path.join(out_dir, "train.csv")
        test_path = os.path.join(out_dir, "test.csv")
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))

        saved_train = pd.read_csv(train_path)
        saved_test = pd.read_csv(test_path)
        self.assertEqual(len(saved_train), 4)
        self.assertEqual(len(saved_test), 2)

    # -------------------- end-to-end (main) --------------------- #
    def test_train_test_split_via_main(self):
        """
        Run the full ingestion `main()` flow by pointing it at a
        temporary params.yaml (via cwd) and confirm splits are produced
        with the expected total row count.
        """
        cwd = os.getcwd()
        try:
            os.chdir(self.tmp_dir)
            # main() looks for "params.yaml" in the current working dir
            # (self.params_path already IS tmp_dir/params.yaml, so no copy needed)
            data_ingestion.main()

            raw_dir = self.params["data_ingestion"]["raw_dir"]
            train_df = pd.read_csv(os.path.join(raw_dir, "train.csv"))
            test_df = pd.read_csv(os.path.join(raw_dir, "test.csv"))

            self.assertEqual(len(train_df) + len(test_df), len(self.sample_df))
        finally:
            os.chdir(cwd)


class TestDataPreprocessing(unittest.TestCase):
    """Unit tests for src/data_preprocessing.py"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        self.raw_df = pd.DataFrame(
            {
                "price": [100000, 200000, 200000, np.nan, 500000],
                "area": [1000, 1500, 1500, 2500, np.nan],
                "mainroad": ["yes", "no", "no", "yes", None],
                "guestroom": ["no", "yes", "yes", "no", "yes"],
            }
        )

        self.binary_cols = ["mainroad", "guestroom"]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------ load_data ------------------------ #
    def test_load_data_success(self):
        csv_path = os.path.join(self.tmp_dir, "train.csv")
        self.raw_df.to_csv(csv_path, index=False)
        df = data_preprocessing.load_data(csv_path)
        self.assertEqual(df.shape, self.raw_df.shape)

    def test_load_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            data_preprocessing.load_data(os.path.join(self.tmp_dir, "missing.csv"))

    # ------------------------ clean_data ------------------------ #
    def test_clean_data_drops_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        cleaned = data_preprocessing.clean_data(df)
        self.assertEqual(len(cleaned), 2)

    def test_clean_data_fills_numeric_na_with_median(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        cleaned = data_preprocessing.clean_data(df)
        self.assertFalse(cleaned["a"].isnull().any())
        # median of [1,2,4] = 2.0
        self.assertEqual(cleaned["a"].iloc[2], 2.0)

    def test_clean_data_fills_categorical_na_with_mode(self):
        # Use an extra distinguishing column so the "yes" rows aren't
        # treated as duplicate rows and dropped before the fillna check.
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "cat": ["yes", "yes", "no", None],
            }
        )
        cleaned = data_preprocessing.clean_data(df)
        self.assertFalse(cleaned["cat"].isnull().any())
        self.assertEqual(cleaned["cat"].iloc[-1], "yes")  # mode is "yes"

    # ----------------------- binary_encode ----------------------- #
    def test_binary_encode_maps_yes_no_to_1_0(self):
        df = pd.DataFrame({"mainroad": ["yes", "no", "yes"]})
        encoded = data_preprocessing.binary_encode(df, ["mainroad"])
        self.assertListEqual(list(encoded["mainroad"]), [1, 0, 1])
        self.assertTrue(pd.api.types.is_integer_dtype(encoded["mainroad"]))

    def test_binary_encode_missing_column_logs_warning_not_raise(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        # Should not raise even though 'mainroad' isn't present
        encoded = data_preprocessing.binary_encode(df, ["mainroad"])
        self.assertNotIn("mainroad", encoded.columns)

    # ------------------------- save_data ------------------------- #
    def test_save_data_creates_file(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        out_path = os.path.join(self.tmp_dir, "nested", "out.csv")
        data_preprocessing.save_data(df, out_path)
        self.assertTrue(os.path.exists(out_path))
        saved = pd.read_csv(out_path)
        self.assertEqual(len(saved), 3)

    # ---------------- full preprocessing pipeline ----------------- #
    def test_full_clean_and_encode_pipeline(self):
        cleaned = data_preprocessing.clean_data(self.raw_df.copy())
        self.assertFalse(cleaned.isnull().any().any())

        encoded = data_preprocessing.binary_encode(cleaned, self.binary_cols)
        for col in self.binary_cols:
            self.assertTrue(set(encoded[col].unique()).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
