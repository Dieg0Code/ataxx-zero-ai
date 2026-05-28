from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from game.actions import ACTION_SPACE
from game.constants import BOARD_SIZE, OBSERVATION_CHANNELS


class AtaxxTransformerNet(nn.Module):
    """Transformer policy-value network for Ataxx."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        value_head_depth: int = 1,
        count_head_enabled: bool = False,
        transformer_pre_ln: bool = True,
        pos_embed_2d: bool = True,
        patch_embed_conv: bool = False,
        num_input_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.board_size = BOARD_SIZE
        self.num_cells = self.board_size * self.board_size
        self.num_actions = ACTION_SPACE.num_actions
        # v15 final: num_input_channels permite a checkpoints pre-v15-final
        # cargar con 11 canales aunque el board produzca 15. Forward recorta
        # x[:, :self.num_input_channels, ...] antes de cualquier proyeccion.
        self.num_input_channels = (
            int(num_input_channels)
            if num_input_channels is not None
            else OBSERVATION_CHANNELS
        )
        self.value_head_depth = int(value_head_depth)
        self.count_head_enabled = bool(count_head_enabled)
        self.transformer_pre_ln = bool(transformer_pre_ln)
        self.pos_embed_2d = bool(pos_embed_2d)
        self.patch_embed_conv = bool(patch_embed_conv)

        # v15 final: patch embedding convolucional opcional. Conv2d 3x3 con
        # padding=1 inyecta inductive bias espacial local (vecindad de cada
        # celda) antes de que el transformer empiece a propagar globalmente.
        # Sin esto el transformer aprende adyacencia 2D desde cero — caro.
        self.input_proj: nn.Module
        self.input_proj_conv: nn.Module
        if self.patch_embed_conv:
            self.input_proj_conv = nn.Conv2d(
                self.num_input_channels,
                d_model,
                kernel_size=3,
                padding=1,
            )
            self.input_proj = nn.Identity()
        else:
            self.input_proj_conv = nn.Identity()
            self.input_proj = nn.Linear(self.num_input_channels, d_model)
        # Two positional schemes:
        # - 1D (legacy <= v14): one learned embedding per (cell + cls), no
        #   geometric prior — the network must learn 2D adjacency from scratch.
        # - 2D (v15+): row_embed[r] + col_embed[c] sumados give a minimal
        #   geometric prior — neighboring cells share half their pos signal.
        if self.pos_embed_2d:
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
            self.row_embed = nn.Parameter(torch.zeros(1, self.board_size, d_model))
            self.col_embed = nn.Parameter(torch.zeros(1, self.board_size, d_model))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_cells + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=self.transformer_pre_ln,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_src_proj = nn.Linear(d_model, d_model // 2)
        self.policy_dst_proj = nn.Linear(d_model, d_model // 2)
        self.policy_scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # value_head_depth=1 (legacy): LN + Linear(d, d) + GELU + Dropout + Linear(d, 1) + Tanh
        # value_head_depth=2 (v11):    LN + Linear(d, 2d) + GELU + LN + Linear(2d, d) + GELU + Dropout + Linear(d, 1) + Tanh
        if self.value_head_depth >= 2:
            self.value_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Tanh(),
            )
        else:
            self.value_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Tanh(),
            )

        # Auxiliary count head: predice diferencia de piezas final desde
        # la perspectiva del jugador que mueve. Solo activa cuando v11.
        if self.count_head_enabled:
            self.count_head: nn.Module = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            self.count_head = nn.Identity()  # placeholder

        self._build_action_cell_indices()
        self._init_weights()

    def _build_action_cell_indices(self) -> None:
        """Precompute board-token indices (src/dst) for each action in ACTION_SPACE."""
        src_list: list[int] = []
        dst_list: list[int] = []
        for action_idx in range(self.num_actions):
            move = ACTION_SPACE.decode(action_idx)
            if move is None:
                # Pass has no squares; we use index 0 and rely on legal-action masking.
                src_list.append(0)
                dst_list.append(0)
                continue
            r1, c1, r2, c2 = move
            src_list.append(r1 * self.board_size + c1)
            dst_list.append(r2 * self.board_size + c2)

        self.register_buffer("_action_src_idx", torch.tensor(src_list, dtype=torch.long))
        self.register_buffer("_action_dst_idx", torch.tensor(dst_list, dtype=torch.long))

    def _init_weights(self) -> None:
        if self.pos_embed_2d:
            nn.init.trunc_normal_(self.cls_pos, std=0.02)
            nn.init.trunc_normal_(self.row_embed, std=0.02)
            nn.init.trunc_normal_(self.col_embed, std=0.02)
        else:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_pos_embed(self) -> torch.Tensor:
        if self.pos_embed_2d:
            # row_embed: (1, 7, d). col_embed: (1, 7, d).
            # Broadcast a (1, 7, 7, d) and sum: each cell gets row[r] + col[c].
            grid = self.row_embed[:, :, None, :] + self.col_embed[:, None, :, :]
            grid = grid.reshape(1, self.num_cells, grid.size(-1))
            return torch.cat([self.cls_pos, grid], dim=1)
        return self.pos_embed

    def build_action_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Derive legal actions from the board observation without target leakage."""
        own = x[:, 0].reshape(x.size(0), self.num_cells)
        empty = x[:, 2].reshape(x.size(0), self.num_cells)
        src_idx = cast(torch.Tensor, self._action_src_idx)
        dst_idx = cast(torch.Tensor, self._action_dst_idx)
        src_is_own = own[:, src_idx] > 0.5
        dst_is_empty = empty[:, dst_idx] > 0.5
        action_mask = (src_is_own & dst_is_empty).to(dtype=x.dtype)

        has_move = torch.any(action_mask[:, : self.num_actions - 1] > 0.0, dim=1)
        action_mask[:, self.num_actions - 1] = (~has_move).to(dtype=x.dtype)
        return action_mask

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, channels, 7, 7]
            action_mask: Optional [batch, num_actions] with 1.0 for legal actions.
        Returns:
            policy_logits: [batch, num_actions]
            value: [batch, 1]
        """
        policy_logits, value, _count = self.forward_with_count(x, action_mask=action_mask)
        return policy_logits, value

    def forward_with_count(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward que devuelve tambien el `count` auxiliar (diff de piezas).

        Si `count_head_enabled=False`, devuelve un tensor de zeros con la
        misma shape que value. Asi el trainer puede activar/desactivar la
        count loss sin ramificar el forward.
        """
        batch_size = x.size(0)
        # v15 final: recortar canales si el modelo fue entrenado con menos
        # de los que produce el board actual (compat para checkpoints viejos).
        if x.size(1) > self.num_input_channels:
            x = x[:, : self.num_input_channels, :, :]
        if self.patch_embed_conv:
            # Conv2d expects (batch, channels, H, W). Output: (batch, d_model, 7, 7).
            x = self.input_proj_conv(x)
            x = x.permute(0, 2, 3, 1).contiguous().view(
                batch_size,
                self.num_cells,
                x.size(1),
            )
        else:
            x = x.permute(0, 2, 3, 1).contiguous().view(
                batch_size,
                self.num_cells,
                self.num_input_channels,
            )
            x = self.input_proj(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, x], dim=1) + self._build_pos_embed()
        encoded = self.encoder(tokens)

        cls_out = encoded[:, 0]
        board_tokens = encoded[:, 1:]

        # Score each move from the semantics of its origin and destination cells.
        src_tokens = board_tokens[:, self._action_src_idx]
        dst_tokens = board_tokens[:, self._action_dst_idx]
        src_feat = self.policy_src_proj(src_tokens)
        dst_feat = self.policy_dst_proj(dst_tokens)
        combined = torch.cat([src_feat, dst_feat], dim=-1)
        policy_logits = self.policy_scorer(combined).squeeze(-1)

        if action_mask is not None:
            min_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(action_mask <= 0, min_value)

        value = self.value_head(cls_out)
        if self.count_head_enabled:
            count = self.count_head(cls_out)
        else:
            count = torch.zeros_like(value)
        return policy_logits, value, count

    def predict(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference helper with softmaxed policy."""
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x, action_mask=action_mask)
            policy = torch.softmax(policy_logits, dim=1)
        return policy, value
