"""Focused tests for MLflow parameter and artifact tracking helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from src import train_models


class MlflowTrackingTests(unittest.TestCase):
    def test_flatten_mlflow_params_serializes_nested_configuration(self) -> None:
        flattened = train_models.flatten_mlflow_params({
            "run": {"seed": 42},
            "features": {"names": ["V1", "Amount"]},
            "models": {"mlp": {"training_rows": None}},
        })

        self.assertEqual(flattened["run.seed"], "42")
        self.assertEqual(flattened["features.names"], '["V1", "Amount"]')
        self.assertEqual(flattened["models.mlp.training_rows"], "None")

    def test_main_uses_one_managed_mlflow_run(self) -> None:
        config = train_models.RunConfig()
        active_run = Mock()
        active_run.info.run_id = "mock-mlflow-run-id"
        run_context = MagicMock()
        run_context.__enter__.return_value = active_run

        with (
            patch.object(train_models.mlflow, "set_experiment") as set_experiment,
            patch.object(train_models.mlflow, "start_run", return_value=run_context) as start_run,
            patch.object(train_models, "_run_training") as run_training,
        ):
            train_models.main(config)

        set_experiment.assert_called_once_with("Bank-Fraud-Detection")
        start_run.assert_called_once_with()
        run_training.assert_called_once_with(config)

    def test_main_propagates_training_failure_through_run_context(self) -> None:
        config = train_models.RunConfig()
        active_run = Mock()
        active_run.info.run_id = "mock-mlflow-run-id"

        class RecordingRunContext:
            exception_type = None

            def __enter__(self):
                return active_run

            def __exit__(self, exception_type, exception, traceback):
                self.exception_type = exception_type
                return False

        run_context = RecordingRunContext()
        with (
            patch.object(train_models.mlflow, "set_experiment"),
            patch.object(train_models.mlflow, "start_run", return_value=run_context),
            patch.object(train_models, "_run_training", side_effect=RuntimeError("synthetic failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                train_models.main(config)

        self.assertIs(run_context.exception_type, RuntimeError)

    def test_sklearn_and_pytorch_artifacts_are_logged_under_model_paths(self) -> None:
        config = train_models.RunConfig()
        with TemporaryDirectory() as temporary_directory:
            for model_name, model_filename in (
                ("random_forest", "random_forest.joblib"),
                ("mlp", "mlp_state_dict.pt"),
            ):
                with self.subTest(model_name=model_name):
                    model_dir = Path(temporary_directory) / model_name
                    model_dir.mkdir()
                    model_path = model_dir / model_filename
                    metadata_path = model_dir / f"{model_name}_metadata.json"
                    model_path.write_bytes(b"synthetic weights")
                    metadata_path.write_text("{}", encoding="utf-8")

                    with patch.object(train_models.mlflow, "log_artifact") as log_artifact:
                        train_models.log_model_artifacts(model_name, model_dir, config)

                    self.assertEqual(log_artifact.call_args_list, [
                        call(str(model_path), artifact_path=f"models/{model_name}"),
                        call(str(metadata_path), artifact_path=f"models/{model_name}"),
                    ])


if __name__ == "__main__":
    unittest.main()
