"""Fail-closed contracts for interface-certified CROMA execution.

ICE is an execution optimization, not a segmentation mechanism.  These
contracts describe exactly which audited backbone representations are consumed
by the downstream receiver.  Unsupported or ambiguous dependencies must fail
closed rather than being approximated.
"""

from __future__ import annotations

from dataclasses import dataclass


class CromaExecutionContractError(RuntimeError):
    """Raised when exact execution equivalence cannot be certified."""


@dataclass(frozen=True)
class BackboneFeatureContract:
    """Representations that the downstream prediction function consumes.

    Stage names are receiver-facing symbolic names (for example ``mid`` and
    ``late``).  ``native_joint`` is deliberately explicit: a caller that starts
    consuming CROMA's native joint encoder must set it to true, which forces the
    execution compiler to retain the full inputs required by that branch.
    """

    optical_stages: tuple[str, ...]
    sar_stages: tuple[str, ...]
    sar_depth_group_stages: tuple[str, ...]
    native_joint: bool
    global_optical: bool = False
    global_sar: bool = False

    def __post_init__(self) -> None:
        for name, values in (
            ("optical_stages", self.optical_stages),
            ("sar_stages", self.sar_stages),
            ("sar_depth_group_stages", self.sar_depth_group_stages),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise CromaExecutionContractError(f"{name} must be a tuple of nonempty stage names")
        if not isinstance(self.native_joint, bool):
            raise CromaExecutionContractError("native_joint must be boolean")
        if not isinstance(self.global_optical, bool) or not isinstance(self.global_sar, bool):
            raise CromaExecutionContractError("global feature flags must be boolean")

    @property
    def stage_union(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.optical_stages, *self.sar_stages, *self.sar_depth_group_stages)
            )
        )
