"""Helpers for loading locomotion policy assets from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


_POLICY_REPO_ID = os.environ.get("UNITREE_MUJOCO_POLICY_REPO", "Artefacts/go2-locomotion")
_POLICY_REVISION = os.environ.get("UNITREE_MUJOCO_POLICY_REVISION")


def _download_policy_file(subfolder: str, filename: str) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=_POLICY_REPO_ID,
                revision=_POLICY_REVISION,
                subfolder=subfolder,
                filename=filename,
                library_name="unitree-mujoco",
            )
        )
    except Exception as exc:
        revision = _POLICY_REVISION or "main"
        raise RuntimeError(
            f"Failed to load policy asset '{subfolder}/{filename}' from "
            f"Hugging Face repo '{_POLICY_REPO_ID}' at revision '{revision}'."
        ) from exc


def get_wtw_assets() -> tuple[Path, Path]:
    cfg_path = _download_policy_file("wtw", "parameters_cpu.pkl")
    _download_policy_file("wtw", "body_latest.jit")
    _download_policy_file("wtw", "adaptation_module_latest.jit")
    return cfg_path.parent, cfg_path


def get_rsl_rl_assets(profile: str = "baseline") -> tuple[Path, Path]:
    subfolder = f"rsl_rl/{profile}"
    env_path = _download_policy_file(subfolder, "env.yaml")
    policy_path = _download_policy_file(subfolder, "policy.pt")
    return env_path, policy_path