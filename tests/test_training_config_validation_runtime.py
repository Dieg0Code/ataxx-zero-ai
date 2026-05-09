from __future__ import annotations

import unittest

from training.config_validation_runtime import validate_bootstrap_warmup_config


def _make_cfg(
    *,
    bootstrap_run_id: str,
    reset_iteration: bool,
    warmup_games: int,
    warmup_epochs: int,
) -> tuple[
    "callable[[str], bool]",
    "callable[[str], int]",
    "callable[[str], str]",
]:
    bools = {"hf_reset_iteration": reset_iteration}
    ints = {"warmup_games": warmup_games, "warmup_epochs": warmup_epochs}
    strs = {"hf_bootstrap_run_id": bootstrap_run_id}
    return bools.__getitem__, ints.__getitem__, strs.__getitem__


class TestValidateBootstrapWarmupConfig(unittest.TestCase):
    def test_no_bootstrap_passes_regardless_of_warmup(self) -> None:
        cfg_bool, cfg_int, cfg_str = _make_cfg(
            bootstrap_run_id="",
            reset_iteration=True,
            warmup_games=0,
            warmup_epochs=0,
        )
        validate_bootstrap_warmup_config(cfg_bool=cfg_bool, cfg_int=cfg_int, cfg_str=cfg_str)

    def test_bootstrap_without_reset_passes(self) -> None:
        cfg_bool, cfg_int, cfg_str = _make_cfg(
            bootstrap_run_id="policy_spatial_v6",
            reset_iteration=False,
            warmup_games=0,
            warmup_epochs=0,
        )
        validate_bootstrap_warmup_config(cfg_bool=cfg_bool, cfg_int=cfg_int, cfg_str=cfg_str)

    def test_bootstrap_with_reset_and_warmup_passes(self) -> None:
        cfg_bool, cfg_int, cfg_str = _make_cfg(
            bootstrap_run_id="policy_spatial_v6",
            reset_iteration=True,
            warmup_games=320,
            warmup_epochs=4,
        )
        validate_bootstrap_warmup_config(cfg_bool=cfg_bool, cfg_int=cfg_int, cfg_str=cfg_str)

    def test_bootstrap_with_reset_and_zero_warmup_games_fails(self) -> None:
        cfg_bool, cfg_int, cfg_str = _make_cfg(
            bootstrap_run_id="policy_spatial_v6",
            reset_iteration=True,
            warmup_games=0,
            warmup_epochs=4,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_bootstrap_warmup_config(cfg_bool=cfg_bool, cfg_int=cfg_int, cfg_str=cfg_str)
        self.assertIn("policy_spatial_v6", str(ctx.exception))
        self.assertIn("warmup_games", str(ctx.exception))

    def test_bootstrap_with_reset_and_zero_warmup_epochs_fails(self) -> None:
        cfg_bool, cfg_int, cfg_str = _make_cfg(
            bootstrap_run_id="policy_spatial_v6",
            reset_iteration=True,
            warmup_games=320,
            warmup_epochs=0,
        )
        with self.assertRaises(ValueError):
            validate_bootstrap_warmup_config(cfg_bool=cfg_bool, cfg_int=cfg_int, cfg_str=cfg_str)


if __name__ == "__main__":
    unittest.main()
