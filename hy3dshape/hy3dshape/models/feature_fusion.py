"""Small, testable tensor adapters used by multi-view image encoders."""

from __future__ import annotations

import torch


def fuse_multiview_features(
    last_hidden_state: torch.Tensor,
    view_blend_weights: torch.Tensor,
) -> torch.Tensor:
    """Blend source-camera patch features into trained camera slots.

    ``last_hidden_state`` has shape ``[batch, source_views, patches, hidden]``
    and weights have shape ``[batch, source_views, output_views]``. Weights are
    normalized defensively so callers cannot accidentally change feature
    magnitude.
    """

    if last_hidden_state.ndim != 4:
        raise ValueError(
            "Multi-view features must have shape [batch, views, patches, hidden]."
        )
    if view_blend_weights.ndim != 3:
        raise ValueError(
            "View blend weights must have shape [batch, source_views, output_views]."
        )
    if tuple(view_blend_weights.shape[:2]) != tuple(last_hidden_state.shape[:2]):
        raise ValueError("View blend weights do not match the encoded source views.")

    weights = view_blend_weights.to(
        device=last_hidden_state.device,
        dtype=last_hidden_state.dtype,
    )
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("View blend weights must be finite and non-negative.")

    totals = weights.sum(dim=1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("Every output camera slot needs a positive weight.")
    normalized = weights / totals
    return torch.einsum(
        "bvph,bvo->boph",
        last_hidden_state,
        normalized,
    )


__all__ = ["fuse_multiview_features"]
