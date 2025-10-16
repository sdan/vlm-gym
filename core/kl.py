"""Simple adaptive KL controller to stabilize updates.

Mirrors the inline logic used in the richer trainer, providing a tiny helper
that adjusts a scalar KL coefficient to keep approx_kl near a target.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveKL:
    target: float = 0.02
    rate: float = 1.5
    coef_min: float = 1e-5
    coef_max: float = 0.5

    def update(self, approx_kl: float, current_coef: float) -> float:
        """Update the KL coefficient given the observed approx_kl.

        - If KL is much larger than target, increase coef multiplicatively.
        - If KL is much smaller than target, decrease coef.
        - Clamp to [coef_min, coef_max].
        """
        coef = float(current_coef)
        if self.target <= 0:
            return max(self.coef_min, min(coef, self.coef_max))

        if approx_kl > self.target * 2.0:
            coef = coef * self.rate if coef > 0 else min(1e-4, self.coef_max)
        elif approx_kl < self.target * 0.5:
            coef = coef / self.rate if coef > 0 else 0.0

        # Hard safety: if KL explodes, escalate once
        if approx_kl > max(0.5, 5.0 * self.target):
            coef = coef * (self.rate ** 2)

        return max(self.coef_min, min(coef, self.coef_max))


__all__ = ["AdaptiveKL"]

