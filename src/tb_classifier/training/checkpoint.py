"""Checkpoint callback that mirrors writes to a second directory.

On Colab the local disk is ephemeral: if the runtime crashes or disconnects,
everything under ``experiments/results`` is lost. :class:`DriveBackupModelCheckpoint`
behaves exactly like a normal :class:`~lightning.pytorch.callbacks.ModelCheckpoint`
but also copies every checkpoint it writes (the top-k *and* ``last.ckpt``) to a
``backup_dir`` such as a mounted Google Drive, and deletes the mirrored copy when
the original is pruned. Resume from any of those files with
``scripts/train.py --resume <path>`` (or ``--resume auto``).

Writes always hit the fast local ``dirpath`` first, then are copied to the backup
via a temp file + atomic ``os.replace`` so a crash mid-copy can never leave a
half-written checkpoint on Drive.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class DriveBackupModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, backup_dir: str | os.PathLike[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup_dir = Path(backup_dir) if backup_dir else None

    def _mirror_path(self, filepath: str) -> Path:
        assert self.backup_dir is not None
        return self.backup_dir / Path(filepath).name

    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if self.backup_dir is None or not trainer.is_global_zero:
            return
        dst = self._mirror_path(filepath)
        tmp = dst.with_name(dst.name + ".tmp")
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filepath, tmp)
            os.replace(tmp, dst)
        except OSError as err:
            rank_zero_warn(f"Drive checkpoint backup failed for {dst}: {err}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _remove_checkpoint(self, trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        if self.backup_dir is None or not trainer.is_global_zero:
            return
        try:
            self._mirror_path(filepath).unlink(missing_ok=True)
        except OSError as err:
            rank_zero_warn(f"Drive checkpoint cleanup failed for {filepath}: {err}")
