"""Proj2/Proj3 Shell 入口必须维持持久化和安全关机契约。"""

from pathlib import Path
import unittest


VERSION_ROOT = Path(__file__).resolve().parents[2]


class EntrypointTests(unittest.TestCase):
    def test_proj2_is_a_real_subset_smoke_test(self) -> None:
        # 文本级契约允许在 Windows 无 Bash/AutoDL 环境验证关键保护措施。
        script = VERSION_ROOT / "Proj2_test_train" / "run.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("make_subset.py", text)
        self.assertIn("--train-list", text)
        self.assertIn("--val-list", text)
        self.assertIn("AUTO_SHUTDOWN", text)
        self.assertIn("tee", text)
        self.assertNotIn('cp -a "${LOCAL_RUN}/."', text)
        self.assertIn("取消自动关机", text)
        self.assertIn('rm -f "${RESULT_DIR}/PASSED"', text)

    def test_proj3_requires_smoke_success_and_runs_full_lists(self) -> None:
        # 正式训练必须依赖冒烟标记，并直接引用完整 split 清单。
        script = VERSION_ROOT / "Proj3_trainning" / "run.sh"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8")
        self.assertIn("Proj2_test_train_v1/PASSED", text)
        self.assertIn("dataset_lists/train_files.txt", text)
        self.assertIn("dataset_lists/val_files.txt", text)
        self.assertIn("AUTO_SHUTDOWN", text)
        self.assertIn("tee", text)
        self.assertNotIn('cp -a "${LOCAL_RUN}/."', text)
        self.assertIn("best_future_distance.pth", text)
        self.assertIn("取消自动关机", text)
        self.assertIn('rm -f "${RESULT_DIR}/PASSED"', text)
        self.assertIn(
            'export OMP_NUM_THREADS="${OPENMP_THREADS}"',
            text,
        )
        self.assertIn(
            "future_delta_head.weight",
            text,
        )

    def test_trainer_reports_successful_early_stopping(self) -> None:
        trainer = (
            VERSION_ROOT / "software" / "see_v1" / "trainer.py"
        ).read_text(encoding="utf-8")

        self.assertIn("FutureDistanceEarlyStopping", trainer)
        self.assertIn("stopper.update(score.future_distance)", trainer)
        self.assertIn('"configured_epochs"', trainer)
        self.assertIn('"epochs_completed"', trainer)
        self.assertIn('"stopped_early"', trainer)
        self.assertIn('"stop_reason"', trainer)


if __name__ == "__main__":
    unittest.main()
