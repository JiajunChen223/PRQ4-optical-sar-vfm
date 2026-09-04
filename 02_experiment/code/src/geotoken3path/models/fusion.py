"""Hard-routed optical--SAR token fusion for the approved GeoToken-3Path route.

The fusion boundary receives coarse optical/SAR tokens with shape ``[B, N, D]``
and an explicit non-spatial SAR depth group with shape ``[B, N, 4, D]``. Every
coarse token is assigned exactly one state: optical bypass, current-scale local
exchange, or depth-group escalation. Only tokens assigned to an active state are dispatched
through that state's operator; bypass tokens leave the fusion boundary exactly
equal to the optical input.

Two-zone cleanup 2026-09-02: all rejected mechanism imports, name registries,
dispatch branches and companion methods were removed (archived in
20_HISTORY/02_legacy_code_pkgs/rejected_mechanisms_20260902/). The verified
baseline surface (always_fuse) and the D1/D2/D3 diagnostic mechanisms remain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from geotoken3path.mechanisms.r2_depth_inject import R2DepthGroupInjector
from geotoken3path.mechanisms.r1_energy_gain import R1LowEnergyChannelGain
from geotoken3path.mechanisms.r3_conditional_depth_select import R3OpticalConditionalDepthSelect
from geotoken3path.mechanisms.r6_dual_channel_inject import R6DualChannelDepthInject
from geotoken3path.mechanisms.r7_residual_upsample import R7ResidualUpsample
from geotoken3path.mechanisms.r8_depth_inject_plus_upsample import R8DepthInjectPlusUpsample
from geotoken3path.mechanisms.r9_optical_semantic_recovery import R9OpticalSemanticRecovery


class GeoToken3PathFusion(nn.Module):
    """Capacity-constrained per-token hard routing with sparse dispatch."""

    BYPASS_STATE = 0
    CURRENT_STATE = 1
    ESCALATION_STATE = 2
    VALID_MECHANISMS = {
        "always_fuse",
        "r2_depth_group_inject",
        "r1_low_energy_channel_gain",
        "r3_optical_conditional_depth_select",
        "r6_depth_dual_channel_inject",
        "r7_residual_learned_upsample",
        "r8_depth_inject_plus_upsample",
        "r9_optical_semantic_recovery",
    }

    def __init__(
        self,
        dim: int,
        active_budget: float = 0.5,
        *,
        local_window_size: int = 7,
        random_route_seed: int = 0,
        evidence_rank: int = 8,
        subpack_candidate_limit: int = 64,
        subpack_edge_budget: int = 32,
    ) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("dim must be at least 4")
        if not 0.0 < active_budget <= 1.0:
            raise ValueError("active_budget must be in (0, 1]")
        if local_window_size < 1 or local_window_size % 2 == 0:
            raise ValueError("local_window_size must be a positive odd integer")
        if isinstance(evidence_rank, bool) or not isinstance(evidence_rank, int) or not 1 <= evidence_rank <= 8:
            raise ValueError("evidence_rank must be an integer in [1, 8]")
        if isinstance(subpack_candidate_limit, bool) or not isinstance(subpack_candidate_limit, int) or not 1 <= subpack_candidate_limit <= 64:
            raise ValueError("subpack_candidate_limit must be an integer in [1, 64]")
        if isinstance(subpack_edge_budget, bool) or not isinstance(subpack_edge_budget, int) or not 1 <= subpack_edge_budget <= subpack_candidate_limit:
            raise ValueError("subpack_edge_budget must be in [1, subpack_candidate_limit]")
        hidden = max(dim // 2, 4)
        self.dim = dim
        self.active_budget = float(active_budget)
        self.local_window_size = int(local_window_size)
        self.random_route_seed = int(random_route_seed)
        self.route_head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.sar_exchange = nn.Linear(dim, dim, bias=False)
        self.sar_escalation = nn.Linear(dim, dim, bias=False)
        # Candidate-only structural alternatives.  Both remain inside the
        # same detector graph and use the same trainability/optimizer policy.
        self.relay_seed = nn.Parameter(torch.zeros(4, dim))
        self.relay_key = nn.Linear(dim, dim, bias=False)
        self.relay_value = nn.Linear(dim, dim, bias=False)
        self.relay_out = nn.Linear(dim, dim, bias=False)
        self.directional_sar = nn.Linear(dim, dim, bias=False)
        self.directional_gate = nn.Sequential(
            nn.Linear(dim * 2, max(dim // 2, 4)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 4), 1),
        )
        self.output_norm = nn.LayerNorm(dim)
        # Fresh-round structural candidates.  These modules are present in the
        # shared detector surface for parity; only the selected mechanism branch
        # invokes them.
        self.edge_query = nn.Linear(dim, dim, bias=False)
        self.edge_key = nn.Linear(dim, dim, bias=False)
        self.edge_value = nn.Linear(dim, dim, bias=False)
        self.edge_reliability = nn.Linear(dim * 2, 1)
        self.transport_offset = nn.Linear(dim * 2, 8)
        self.transport_value = nn.Linear(dim, dim, bias=False)
        self.group_router = nn.Linear(dim * 2, 6)
        self.group_value = nn.Linear(dim, dim, bias=False)
        self.ot_query = nn.Linear(dim, dim, bias=False)
        self.ot_key = nn.Linear(dim, dim, bias=False)
        self.ot_value = nn.Linear(dim, dim, bias=False)
        self.ot_null_cost = nn.Linear(dim * 2, 1)
        self.ot_out = nn.Linear(dim, dim, bias=False)
        self.triad_optical = nn.Linear(dim, dim, bias=False)
        self.triad_sar = nn.Linear(dim, dim, bias=False)
        self.triad_depth = nn.Linear(dim, dim, bias=False)
        self.triad_out = nn.Linear(dim, dim, bias=False)
        self.mass_coarse = nn.Linear(dim * 2, 1)
        self.mass_fine = nn.Linear(dim * 2, 1)
        self.warm_relation = nn.Sequential(nn.Linear(dim * 2, max(dim // 2, 4)), nn.GELU(), nn.Linear(max(dim // 2, 4), dim))
        self.boundary_gate = nn.Linear(dim * 2, 1)
        self.boundary_out = nn.Linear(dim, dim, bias=False)
        self.calibration_gate = nn.Linear(dim * 2, dim)
        self.calibration_out = nn.Linear(dim, dim, bias=False)
        self.depth_state_update = nn.Sequential(nn.Linear(dim * 2, max(dim // 2, 4)), nn.GELU(), nn.Linear(max(dim // 2, 4), dim))
        self.depth_state_gate = nn.Linear(dim * 2, 1)
        self.depth_state_out = nn.Linear(dim, dim, bias=False)
        self.stage_bridge = nn.Sequential(nn.Linear(dim * 2, max(dim // 2, 4)), nn.GELU(), nn.Linear(max(dim // 2, 4), dim))
        # CEAK successor operators.  These parameters are allocated for every
        # mechanism so baseline/candidate state-dict and trainability surfaces
        # remain identical.  Zero-start scales make each candidate an exact
        # baseline-preserving residual until it is explicitly enabled.
        self.ceak_rank = int(evidence_rank)
        self.subpack_candidate_limit = int(subpack_candidate_limit)
        self.subpack_edge_budget = int(subpack_edge_budget)
        self.ceak_optical_evidence = nn.Linear(dim, self.ceak_rank, bias=False)
        self.ceak_sar_evidence = nn.Linear(dim, self.ceak_rank, bias=False)
        self.ceak_query = nn.Linear(dim, dim, bias=False)
        self.ceak_key = nn.Linear(dim, dim, bias=False)
        self.ceak_value = nn.Linear(dim, dim, bias=False)
        self.ceak_private = nn.Linear(dim, dim, bias=False)
        self.ceak_null = nn.Linear(dim * 2, 1)
        self.ceak_scale = nn.Parameter(torch.zeros(1))
        self.subpack_scale = nn.Parameter(torch.zeros(1))
        self.cfedge_scale = nn.Parameter(torch.zeros(1))
        self.cfedge_utility = nn.Sequential(
            # Coalition utilities operate on compact rank-8 evidence packets,
            # not expanded D-dimensional edge tensors, keeping the four
            # coalition pass bounded on the target 24GB card.
            nn.Linear(self.ceak_rank * 2, max(dim // 2, 4)),
            nn.GELU(),
            nn.Linear(max(dim // 2, 4), 1),
        )
        self.cfedge_private = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.warm_relation[-1].weight)
        nn.init.zeros_(self.warm_relation[-1].bias)
        nn.init.zeros_(self.boundary_out.weight)
        nn.init.zeros_(self.calibration_out.weight)
        nn.init.zeros_(self.depth_state_update[-1].weight)
        nn.init.zeros_(self.depth_state_update[-1].bias)
        nn.init.zeros_(self.depth_state_out.weight)
        nn.init.zeros_(self.stage_bridge[-1].weight)
        nn.init.zeros_(self.stage_bridge[-1].bias)

    def _budget_count(self, tokens: int) -> int:
        """Return the exact active-token capacity used by hard routes."""

        if tokens < 1:
            raise ValueError("token dimension must be non-empty")
        count = int(math.floor(self.active_budget * tokens + 0.5))
        return min(tokens, max(1, count))

    @staticmethod
    def _empty_hard_routes(optical: Tensor, state: int) -> Tensor:
        batch, tokens, _ = optical.shape
        return torch.full(
            (batch, tokens),
            state,
            dtype=torch.long,
            device=optical.device,
        )

    def _deterministic_budget_mask(self, optical: Tensor) -> Tensor:
        """Select an evenly spaced, exact-capacity token set per sample."""

        batch, tokens, _ = optical.shape
        count = self._budget_count(tokens)
        positions = torch.floor(
            (torch.arange(count, device=optical.device, dtype=torch.float32) + 0.5)
            * (tokens / count)
        ).long()
        mask = torch.zeros((batch, tokens), dtype=torch.bool, device=optical.device)
        mask[:, positions] = True
        return mask

    def _random_budget_routes(self, optical: Tensor) -> Tensor:
        """Create repeatable random hard routes with an exact active capacity."""

        batch, tokens, _ = optical.shape
        count = self._budget_count(tokens)
        routes = self._empty_hard_routes(optical, self.BYPASS_STATE)
        # Generate indices on CPU for device-independent seed semantics.
        for batch_index in range(batch):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.random_route_seed + batch_index)
            active = torch.randperm(tokens, generator=generator)[:count]
            active_states = torch.randint(
                self.CURRENT_STATE,
                self.ESCALATION_STATE + 1,
                (count,),
                generator=generator,
            )
            routes[batch_index, active.to(optical.device)] = active_states.to(optical.device)
        return routes

    def _candidate_routes(self, route_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Return exact-capacity hard routes and a straight-through estimate."""

        soft_routes = route_logits.softmax(dim=-1)
        batch, tokens, _ = soft_routes.shape
        count = self._budget_count(tokens)
        hard_states = torch.full(
            (batch, tokens),
            self.BYPASS_STATE,
            dtype=torch.long,
            device=route_logits.device,
        )
        active_score = 1.0 - soft_routes[..., self.BYPASS_STATE]
        active_indices = active_score.topk(count, dim=1, largest=True, sorted=True).indices
        active_preferences = soft_routes[..., self.CURRENT_STATE : self.ESCALATION_STATE + 1]
        selected_preferences = active_preferences.gather(
            1,
            active_indices.unsqueeze(-1).expand(-1, -1, 2),
        )
        selected_states = selected_preferences.argmax(dim=-1) + self.CURRENT_STATE
        hard_states.scatter_(1, active_indices, selected_states)
        hard_one_hot = F.one_hot(hard_states, num_classes=3).to(route_logits.dtype)
        # Forward values are one-hot; backward gradients follow the soft router.
        # Parentheses make the forward correction exactly zero (rather than
        # ``(hard + soft) - soft``), preserving exact one-hot values.
        straight_through = hard_one_hot + (soft_routes - soft_routes.detach())
        return hard_states, straight_through

    def _local_context_selected(
        self,
        optical: Tensor,
        sar: Tensor,
        selected: Tensor,
        *,
        window_size: int,
    ) -> Tensor:
        """Compute local SAR attention only for selected ``[batch, token]`` rows."""

        if selected.numel() == 0:
            return optical.new_empty((0, self.dim))
        _, tokens, _ = sar.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("current-scale local exchange requires a square token grid")
        batch_index, token_index = selected.unbind(dim=1)
        row = token_index // side
        col = token_index % side
        radius = window_size // 2
        offsets = torch.arange(-radius, radius + 1, device=sar.device)
        delta_row, delta_col = torch.meshgrid(offsets, offsets, indexing="ij")
        delta_row = delta_row.reshape(1, -1)
        delta_col = delta_col.reshape(1, -1)
        neighbor_row = row.unsqueeze(1) + delta_row
        neighbor_col = col.unsqueeze(1) + delta_col
        valid = (
            (neighbor_row >= 0)
            & (neighbor_row < side)
            & (neighbor_col >= 0)
            & (neighbor_col < side)
        )
        neighbor_index = neighbor_row.clamp(0, side - 1) * side + neighbor_col.clamp(0, side - 1)
        neighborhoods = sar[batch_index.unsqueeze(1), neighbor_index]
        projected = self.sar_exchange(neighborhoods)
        query = optical[batch_index, token_index].unsqueeze(1)
        scores = (query * projected).sum(dim=-1) / math.sqrt(self.dim)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * projected).sum(dim=1)

    def _depth_group_context_selected(
        self,
        optical: Tensor,
        depth_group: Tensor,
        selected: Tensor,
    ) -> Tensor:
        """Attend over one explicit four-depth SAR group per selected token."""

        if selected.numel() == 0:
            return optical.new_empty((0, self.dim))
        batch, tokens, _ = optical.shape
        if depth_group.ndim == 3:
            if depth_group.shape != (batch, tokens * 4, self.dim):
                raise ValueError("flattened depth_group must have shape [batch, tokens*4, dim]")
            depth_blocks = depth_group.reshape(batch, tokens, 4, self.dim)
        elif depth_group.ndim == 4:
            if depth_group.shape != (batch, tokens, 4, self.dim):
                raise ValueError("depth_group must have shape [batch, tokens, 4, dim]")
            depth_blocks = depth_group
        else:
            raise ValueError("depth_group must be a [B, N, 4, D] block tensor")
        batch_index, token_index = selected.unbind(dim=1)
        blocks = depth_blocks[batch_index, token_index]
        projected = self.sar_escalation(blocks)
        query = optical[batch_index, token_index].unsqueeze(1)
        scores = (query * projected).sum(dim=-1) / math.sqrt(self.dim)
        weights = scores.softmax(dim=-1)
        return (weights.unsqueeze(-1) * projected).sum(dim=1)

    def _relay_context_selected(
        self,
        optical: Tensor,
        sar: Tensor,
        selected: Tensor,
    ) -> Tensor:
        """Route selected optical rows through a fixed four-slot SAR relay.

        The relay is intentionally small and global over the token set: SAR
        tokens write to four learned slots, then selected optical tokens read
        from those slots.  This is a distinct information path from local
        same-index exchange and remains bounded for the 3090 rapid screen.
        """

        if selected.numel() == 0:
            return optical.new_empty((0, self.dim))
        batch, _, _ = sar.shape
        relay = self.relay_seed.unsqueeze(0).expand(batch, -1, -1)
        keys = self.relay_key(relay)
        write_scores = torch.matmul(sar, keys.transpose(1, 2)) / math.sqrt(self.dim)
        write_weights = write_scores.transpose(1, 2).softmax(dim=-1)
        values = self.relay_value(sar)
        relay = relay + torch.matmul(write_weights, values)
        batch_index, token_index = selected.unbind(dim=1)
        queries = optical[batch_index, token_index]
        read_scores = torch.matmul(queries.unsqueeze(1), relay[batch_index].transpose(1, 2)).squeeze(1)
        read_weights = read_scores.softmax(dim=-1)
        context = torch.bmm(read_weights.unsqueeze(1), relay[batch_index]).squeeze(1)
        return self.relay_out(context)

    def _reliability_context_selected(
        self,
        optical: Tensor,
        sar: Tensor,
        selected: Tensor,
        *,
        window_size: int,
    ) -> tuple[Tensor, Tensor]:
        """Select directed local SAR edges using feature disagreement.

        The node set is preserved; only the local correspondence edges are
        sparsified.  ``edge_entropy`` is returned as an audit observable.
        """

        if selected.numel() == 0:
            return optical.new_empty((0, self.dim)), optical.new_empty((0,))
        _, tokens, _ = sar.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError("reliability edge transport requires a square token grid")
        batch_index, token_index = selected.unbind(dim=1)
        row = token_index // side
        col = token_index % side
        radius = window_size // 2
        offsets = torch.arange(-radius, radius + 1, device=sar.device)
        delta_row, delta_col = torch.meshgrid(offsets, offsets, indexing="ij")
        delta_row = delta_row.reshape(1, -1)
        delta_col = delta_col.reshape(1, -1)
        neighbor_row = row.unsqueeze(1) + delta_row
        neighbor_col = col.unsqueeze(1) + delta_col
        valid = (
            (neighbor_row >= 0)
            & (neighbor_row < side)
            & (neighbor_col >= 0)
            & (neighbor_col < side)
        )
        neighbor_index = neighbor_row.clamp(0, side - 1) * side + neighbor_col.clamp(0, side - 1)
        neighborhoods = sar[batch_index.unsqueeze(1), neighbor_index]
        queries = optical[batch_index, token_index].unsqueeze(1)
        projected_keys = self.edge_key(neighborhoods)
        projected_values = self.edge_value(neighborhoods)
        disagreement = (queries - neighborhoods).abs().mean(dim=-1, keepdim=True)
        reliability_penalty = torch.sigmoid(
            self.edge_reliability(torch.cat([queries.expand_as(neighborhoods), neighborhoods], dim=-1))
        ).squeeze(-1)
        scores = (self.edge_query(queries) * projected_keys).sum(dim=-1) / math.sqrt(self.dim)
        scores = scores - disagreement.squeeze(-1) - reliability_penalty
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        context = (weights.unsqueeze(-1) * projected_values).sum(dim=1)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        return context, entropy

    def _validate_successor_tokens(self, optical: Tensor, sar: Tensor) -> None:
        """Fail closed on the token contract used by the CEAK successor."""

        if optical.ndim != 3 or sar.ndim != 3 or optical.shape != sar.shape:
            raise ValueError("CEAK operators require matching [B,N,D] token tensors")
        if optical.shape[-1] != self.dim or optical.shape[1] < 1:
            raise ValueError("CEAK token dimension must be non-empty and equal to dim")
        if not torch.isfinite(optical).all() or not torch.isfinite(sar).all():
            raise ValueError("CEAK operators reject non-finite token inputs")

    @staticmethod
    def _zero_start_residual(value: Tensor, scale: Tensor) -> Tensor:
        """Return a value-scaled residual with a live initial gradient.

        ``value - value.detach()`` is exactly zero in the forward pass but has
        unit derivative with respect to the mechanism path.  The explicit
        straight-through term therefore preserves bitwise baseline identity at
        scale zero without freezing every upstream evidence parameter on the
        first optimizer step.
        """

        scaled = value * scale.to(value.dtype)
        return scaled + (value - value.detach())

    def _successor_edge_terms(self, optical: Tensor, sar: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return edge logits, conflict, SAR values and evidence packets.

        Evidence heads are normalized positive rank-8 distributions.  The
        pairwise dot product is a shared-evidence score; its complement is an
        explicit conflict quantity used by all three successor mechanisms.
        """

        self._validate_successor_tokens(optical, sar)
        with torch.autocast(device_type=optical.device.type, enabled=False):
            optical_fp = optical.float()
            sar_fp = sar.float()
            optical_evidence = F.softplus(self.ceak_optical_evidence(optical_fp)) + 1e-6
            sar_evidence = F.softplus(self.ceak_sar_evidence(sar_fp)) + 1e-6
            optical_evidence = optical_evidence / optical_evidence.sum(dim=-1, keepdim=True)
            sar_evidence = sar_evidence / sar_evidence.sum(dim=-1, keepdim=True)
            shared = torch.matmul(optical_evidence, sar_evidence.transpose(-1, -2))
            conflict = (1.0 - shared).clamp(0.0, 1.0)
            query = self.ceak_query(optical_fp)
            key = self.ceak_key(sar_fp)
            edge_logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.dim)
            edge_logits = edge_logits - conflict
            values = self.ceak_value(sar_fp)
            packets = sar_evidence
        return edge_logits, conflict, values, packets

    def _subpack_select(self, packets: Tensor, quality: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Greedy hard edge-packet selection with a log-det gain surrogate."""

        batch, tokens, candidates, rank = packets.shape
        query_count = batch * tokens
        packet_rows = F.normalize(packets.reshape(query_count, candidates, rank), dim=-1)
        quality_rows = quality.reshape(query_count, candidates).clamp_min(1e-6)
        chosen = torch.zeros((query_count, candidates), dtype=torch.bool, device=packets.device)
        basis = torch.zeros((query_count, k, rank), dtype=packet_rows.dtype, device=packets.device)
        selected_indices = []
        gains = []
        for step in range(k):
            residual = packet_rows
            if step:
                coefficients = torch.einsum("qmr,qkr->qmk", packet_rows, basis[:, :step])
                residual = packet_rows - torch.einsum("qmk,qkr->qmr", coefficients, basis[:, :step])
            residual_norm = residual.square().sum(dim=-1)
            utility = torch.log(quality_rows + 1e-6) + torch.log(residual_norm + 1e-6)
            utility = utility.masked_fill(chosen, torch.finfo(utility.dtype).min)
            index = utility.argmax(dim=-1)
            selected_indices.append(index)
            gains.append(utility.gather(1, index.unsqueeze(1)).squeeze(1))
            chosen.scatter_(1, index.unsqueeze(1), True)
            selected = packet_rows.gather(1, index[:, None, None].expand(-1, 1, rank)).squeeze(1)
            basis[:, step] = F.normalize(selected, dim=-1)
        return chosen.reshape(batch, tokens, candidates), torch.stack(gains, dim=-1).reshape(batch, tokens, k)

    def _dispatch(
        self,
        optical: Tensor,
        sar: Tensor,
        depth_group: Tensor | None,
        hard_states: Tensor,
        route_estimate: Tensor,
        *,
        current_window_size: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Dispatch only selected tokens and scatter residuals to the optical path."""

        current_selected = (hard_states == self.CURRENT_STATE).nonzero(as_tuple=False)
        escalation_selected = (hard_states == self.ESCALATION_STATE).nonzero(as_tuple=False)
        residual = torch.zeros_like(optical)

        if current_selected.numel() > 0:
            current_delta = self._local_context_selected(
                optical,
                sar,
                current_selected,
                window_size=current_window_size,
            )
            current_delta = self.output_norm(current_delta)
            current_batch, current_token = current_selected.unbind(dim=1)
            current_gate = route_estimate[
                current_batch,
                current_token,
                self.CURRENT_STATE,
            ].unsqueeze(-1)
            residual = residual.index_put(
                (current_batch, current_token),
                current_delta * current_gate,
                accumulate=False,
            )

        if escalation_selected.numel() > 0:
            if depth_group is None:
                raise ValueError("depth-group escalation requires explicit SAR depth-group tokens")
            escalation_delta = self._depth_group_context_selected(
                optical,
                depth_group,
                escalation_selected,
            )
            escalation_delta = self.output_norm(escalation_delta)
            escalation_batch, escalation_token = escalation_selected.unbind(dim=1)
            escalation_gate = route_estimate[
                escalation_batch,
                escalation_token,
                self.ESCALATION_STATE,
            ].unsqueeze(-1)
            residual = residual.index_put(
                (escalation_batch, escalation_token),
                escalation_delta * escalation_gate,
                accumulate=False,
            )

        # The selected-state gate is one in the forward pass and carries the
        # straight-through router gradient in the backward pass. Bypass rows
        # therefore remain bitwise equal to optical while still training the
        # decision that selected the bypass state.
        selected_gate = route_estimate.gather(
            -1,
            hard_states.unsqueeze(-1),
        ).squeeze(-1)
        fused = optical * selected_gate.unsqueeze(-1) + residual
        counts = torch.stack(
            [
                (hard_states == self.BYPASS_STATE).sum(),
                (hard_states == self.CURRENT_STATE).sum(),
                (hard_states == self.ESCALATION_STATE).sum(),
            ]
        )
        return fused, {
            "branch_counts": counts,
            "bypass_count": counts[0],
            "current_count": counts[1],
            "escalation_count": counts[2],
            "active_count": counts[1] + counts[2],
        }

    def forward(
        self,
        optical: Tensor,
        sar: Tensor,
        *,
        mechanism_set: str,
        depth_group: Tensor | None = None,
        physical_groups: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if mechanism_set not in self.VALID_MECHANISMS:
            raise ValueError(f"unknown mechanism_set: {mechanism_set}")
        if optical.ndim != 3 or sar.ndim != 3:
            raise ValueError("optical and sar must have shape [batch, tokens, dim]")
        if optical.shape != sar.shape:
            raise ValueError("optical and sar token shapes must match")
        if optical.shape[-1] != self.dim:
            raise ValueError(f"expected token dim {self.dim}, received {optical.shape[-1]}")

        route_logits: Tensor | None = None
        if mechanism_set == "always_fuse":
            hard_states = self._empty_hard_routes(optical, self.CURRENT_STATE)
            route_estimate = F.one_hot(hard_states, num_classes=3).to(optical.dtype)
        else:
            route_logits = self.route_head(torch.cat([optical, sar], dim=-1))
            hard_states, route_estimate = self._candidate_routes(route_logits)

        # Conventional always-fuse uses same-index exchange; the other current
        # controls and candidate use the approved local-window operator.
        current_window_size = 1 if mechanism_set == "always_fuse" else self.local_window_size
        fused, dispatch_aux = self._dispatch(
            optical,
            sar,
            depth_group,
            hard_states,
            route_estimate,
            current_window_size=current_window_size,
        )
        hard_one_hot = F.one_hot(hard_states, num_classes=3).to(optical.dtype)
        total_tokens = hard_states.numel()
        soft_route_probs = (
            route_estimate.detach()
            if route_logits is None
            else route_logits.softmax(dim=-1)
        )
        aux: dict[str, Tensor] = {
            "route_probs": route_estimate,
            "soft_route_probs": soft_route_probs,
            "hard_route": hard_states,
            "hard_route_one_hot": hard_one_hot,
            "active_fraction": dispatch_aux["active_count"].to(optical.dtype) / total_tokens,
            **dispatch_aux,
        }
        return fused, aux


class OpticalSarTokenModel(nn.Module):
    """Shared staged fusion and segmentation surface for every mechanism set."""

    def __init__(
        self,
        *,
        dim: int = 32,
        num_classes: int = 19,
        active_budget: float = 0.5,
        mechanism_set: str = "always_fuse",
        local_window_size: int = 7,
        stages: Sequence[str] = ("mid", "late"),
        allow_synthetic_depth_group_fallback: bool = False,
    ) -> None:
        super().__init__()
        if mechanism_set not in GeoToken3PathFusion.VALID_MECHANISMS:
            raise ValueError(f"unknown mechanism_set: {mechanism_set}")
        self.dim = dim
        self.num_classes = num_classes
        self.mechanism_set = mechanism_set
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("stages must be a non-empty unique sequence")
        self.stages = tuple(str(stage) for stage in stages)
        self.allow_synthetic_depth_group_fallback = bool(allow_synthetic_depth_group_fallback)
        self.optical_stem = nn.Linear(dim, dim)
        self.sar_stem = nn.Linear(dim, dim)
        self.fusions = nn.ModuleDict(
            {
                stage: GeoToken3PathFusion(
                    dim,
                    active_budget=active_budget,
                    local_window_size=local_window_size,
                )
                for stage in self.stages
            }
        )
        self.classifier = nn.Linear(dim, num_classes)
        mechanism_id = mechanism_set
        if mechanism_id == "r2_depth_group_inject":
            self.router = R2DepthGroupInjector(dim)
        elif mechanism_id == "r1_low_energy_channel_gain":
            self.router = R1LowEnergyChannelGain()
        elif mechanism_id == "r3_optical_conditional_depth_select":
            self.router = R3OpticalConditionalDepthSelect(dim, tuple(self.stages))
        elif mechanism_id == "r6_depth_dual_channel_inject":
            self.router = R6DualChannelDepthInject(dim, tuple(self.stages))
        elif mechanism_id == "r7_residual_learned_upsample":
            self.router = R7ResidualUpsample(num_classes)
        elif mechanism_id == "r8_depth_inject_plus_upsample":
            self.router = R8DepthInjectPlusUpsample(dim, num_classes, tuple(self.stages))
        elif mechanism_id == "r9_optical_semantic_recovery":
            self.router = R9OpticalSemanticRecovery(dim)
        else:
            self.router = None
        self.stage_bridge = nn.Sequential(nn.Linear(dim * 2, max(dim // 2, 4)), nn.GELU(), nn.Linear(max(dim // 2, 4), dim))
        nn.init.zeros_(self.stage_bridge[-1].weight)
        nn.init.zeros_(self.stage_bridge[-1].bias)
        # Unconditional freeze: physical-group projections belong to a rejected
        # mechanism and are inert shared-surface capacity only.
        self.physical_group_channels = (4, 4, 2, 2, 1, 1)
        self.physical_group_projections = nn.ModuleList(
            [nn.Linear(channels, dim) for channels in self.physical_group_channels]
        )

    @property
    def fusion(self) -> GeoToken3PathFusion:
        """Compatibility view of the first stage for focused unit tests."""

        return self.fusions[self.stages[0]]

    @staticmethod
    def _synthetic_depth_group(sar: Tensor) -> Tensor:
        """Create an explicitly synthetic repeated depth group for smoke tests only."""

        batch, tokens, dim = sar.shape
        return sar.unsqueeze(2).expand(batch, tokens, 4, dim)

    @staticmethod
    def _stage_tensor(value: Tensor | Mapping[str, Tensor], stage: str, name: str) -> Tensor:
        if isinstance(value, Tensor):
            return value
        if not isinstance(value, Mapping) or stage not in value:
            raise ValueError(f"{name} is missing stage {stage}")
        tensor = value[stage]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name}[{stage}] must be a tensor")
        return tensor

    def project_physical_groups(
        self,
        optical_image: Tensor,
        sar_image: Tensor,
        *,
        token_count: int,
    ) -> Tensor:
        """Create six auditable physical-group token streams from raw inputs."""

        side = math.isqrt(int(token_count))
        if side * side != int(token_count):
            raise ValueError("physical group dispatch requires a square token grid")
        optical_groups = (
            optical_image[:, 0:4],
            optical_image[:, 4:8],
            optical_image[:, 8:10],
            optical_image[:, 10:12],
        )
        sar_groups = (sar_image[:, 0:1], sar_image[:, 1:2])
        groups = optical_groups + sar_groups
        outputs = []
        for projection, group in zip(self.physical_group_projections, groups):
            pooled = F.adaptive_avg_pool2d(group, (side, side)).flatten(2).transpose(1, 2)
            outputs.append(projection(pooled))
        return torch.stack(outputs, dim=2)

    def forward(
        self,
        optical: Tensor | Mapping[str, Tensor],
        sar: Tensor | Mapping[str, Tensor],
        *,
        joint: Tensor | Mapping[str, Tensor] | None = None,
        depth_group: Tensor | Mapping[str, Tensor] | None = None,
        physical_groups: Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        depth_group_input_provided = depth_group is not None
        fused: Tensor | None = None
        stage_aux: dict[str, dict[str, Tensor]] = {}
        for stage in self.stages:
            raw_optical_stage = self._stage_tensor(optical, stage, "optical")
            raw_sar_stage = self._stage_tensor(sar, stage, "sar")
            pure_optical_stage = self.optical_stem(raw_optical_stage)
            pure_sar_stage = self.sar_stem(raw_sar_stage)
            optical_stage = pure_optical_stage
            sar_stage = pure_sar_stage
            if fused is not None:
                if fused.shape != optical_stage.shape:
                    raise ValueError("successive stage token shapes must match in the local bridge")
                optical_stage = optical_stage + fused
            depth_features = (
                None
                if depth_group is None
                else self.sar_stem(self._stage_tensor(depth_group, stage, "depth_group"))
            )
            if (
                self.mechanism_set == "r2_depth_group_inject"
                and stage == self.stages[-1]
                and depth_features is not None
                and self.router is not None
            ):
                sar_stage = self.router(depth_features, sar_stage)
            if self.mechanism_set == "r3_optical_conditional_depth_select" and depth_features is not None and self.router is not None:
                sar_stage = self.router(depth_features, sar_stage, optical_stage, stage)
            if (
                self.mechanism_set == "r8_depth_inject_plus_upsample"
                and depth_features is not None
                and self.router is not None
            ):
                sar_stage = self.router.inject_depth(depth_features, sar_stage, optical_stage, stage)
            if (
                self.mechanism_set == "r6_depth_dual_channel_inject"
                and depth_features is not None
                and self.router is not None
            ):
                optical_stage, sar_stage = self.router(
                    depth_features, optical_stage, sar_stage, stage
                )
            if (
                self.mechanism_set == "r9_optical_semantic_recovery"
                and stage == self.stages[-1]
                and self.router is not None
            ):
                optical_stage = self.router(optical_stage)
            # External mechanisms (R2/R1) act outside the fusion boundary; the
            # fusion layer itself always executes the verified always-fuse path.
            stage_mechanism = (
                "always_fuse"
                if self.mechanism_set
                in {"r2_depth_group_inject", "r1_low_energy_channel_gain",
                    "r3_optical_conditional_depth_select", "r6_depth_dual_channel_inject",
                    "r7_residual_learned_upsample", "r8_depth_inject_plus_upsample",
                    "r9_optical_semantic_recovery"}
                else self.mechanism_set
            )
            fused, one_stage_aux = self.fusions[stage](
                optical_stage,
                sar_stage,
                mechanism_set=stage_mechanism,
                depth_group=depth_features,
                physical_groups=physical_groups,
            )
            stage_aux[stage] = one_stage_aux

        assert fused is not None
        last_aux = stage_aux[self.stages[-1]]
        aux: dict[str, Any] = {**last_aux, "stages": stage_aux}
        # Raw-image decoder mechanisms may consume the final fused carrier.
        # It remains part of the same end-to-end graph and is never detached.
        aux["final_fused_tokens"] = fused
        reference = self._stage_tensor(optical, self.stages[-1], "optical")
        aux["depth_group_input_provided"] = torch.tensor(
            depth_group_input_provided,
            device=reference.device,
            dtype=torch.bool,
        )
        if self.mechanism_set == "r1_low_energy_channel_gain" and self.router is not None:
            fused = self.router(fused)
        logits = self.classifier(fused)
        if output_size is not None:
            side = math.isqrt(logits.shape[1])
            if side * side != logits.shape[1]:
                raise ValueError("dense segmentation output requires a square token grid")
            logits = logits.transpose(1, 2).reshape(logits.shape[0], self.num_classes, side, side)
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
            if self.mechanism_set == "r7_residual_learned_upsample" and self.router is not None:
                logits = self.router(logits)
            if self.mechanism_set == "r8_depth_inject_plus_upsample" and self.router is not None:
                logits = self.router.refine_logits(logits)
        if return_aux:
            return logits, aux
        return logits
