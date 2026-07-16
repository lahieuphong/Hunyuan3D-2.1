# -*- coding: utf-8 -*-

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only


class PeftSaveCallback(Callback):
    """Save only the trainable PEFT adapter instead of the frozen base model."""

    def __init__(
        self,
        save_dir: str,
        save_every_n_steps: Optional[int] = None,
        save_on_train_epoch_end: bool = False,
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self._last_saved_step = -1

        if self.save_every_n_steps is not None and self.save_every_n_steps <= 0:
            raise ValueError("save_every_n_steps must be greater than zero or None")

    @staticmethod
    def _get_peft_model(pl_module):
        peft_model = getattr(pl_module, "model", None)
        if peft_model is None or not hasattr(peft_model, "peft_config"):
            raise RuntimeError(
                "PeftSaveCallback requires pl_module.model to be a PEFT model. "
                "Enable model.params.lora_config in the training config."
            )
        return peft_model

    @rank_zero_only
    def _save(self, pl_module, folder_name: str, global_step: int) -> None:
        if global_step == self._last_saved_step and folder_name != "final":
            return

        peft_model = self._get_peft_model(pl_module)
        save_path = self.save_dir / folder_name
        save_path.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(save_path), safe_serialization=True)
        self._last_saved_step = global_step
        rank_zero_info(f"[PeftSaveCallback] Saved LoRA adapter to {save_path}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.save_every_n_steps is None:
            return

        global_step = trainer.global_step
        if global_step > 0 and global_step % self.save_every_n_steps == 0:
            self._save(pl_module, f"step_{global_step:08d}", global_step)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.save_on_train_epoch_end:
            self._save(
                pl_module,
                f"epoch_{trainer.current_epoch:06d}",
                trainer.global_step,
            )

    def on_train_end(self, trainer, pl_module) -> None:
        self._save(pl_module, "final", trainer.global_step)
