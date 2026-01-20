"""Slim multimodal fusion model with visual self-attention, cross-attention,
and adaptive goal conditioning.

Architecture summary:
- Project per-modality features to a common dimension d_model.
- Self-attend over the visual stream to form temporal context.
- Cross-attend from the visual context to other modalities (pose, motion, gaze)
  and sum their attention contributions to enrich visual context.
- Optionally apply adaptive goal conditioning: compute a per-timestep gate Uhat
  from fused features and mix fused features with goal embeddings:
    h_tilde = (1 - Uhat) * h_fuse + Uhat * h_goal
- Pool and classify; optional small future-prediction head.

This implements the pattern you described: visual self-attention -> cross-attend
to other modalities -> adaptive goal conditioning.
"""
from typing import Optional

import torch
import torch.nn as nn


class SlimFusionTransformer(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        pose_dim: int = 3,
        gaze_dim: int = 0,
        motion_dim: int = 2,
        goal_dim: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_classes: int = 5,
        max_time_steps: int = 64,
        predict_future_steps: int = 0,
    ):
        super().__init__()

        # config
        self.visual_dim = int(visual_dim)
        self.pose_dim = int(pose_dim)
        self.gaze_dim = int(gaze_dim)
        self.motion_dim = int(motion_dim)
        self.goal_dim = int(goal_dim)
        self.d_model = int(d_model)
        self.max_time_steps = int(max_time_steps)

        # per-modality projections
        self.visual_proj = nn.Linear(self.visual_dim, self.d_model) if self.visual_dim > 0 else None
        self.pose_proj = nn.Linear(self.pose_dim, self.d_model) if self.pose_dim > 0 else None
        self.gaze_proj = nn.Linear(self.gaze_dim, self.d_model) if self.gaze_dim > 0 else None
        self.motion_proj = nn.Linear(self.motion_dim, self.d_model) if self.motion_dim > 0 else None
        self.goal_proj = nn.Linear(self.goal_dim, self.d_model) if self.goal_dim > 0 else None

        # learned per-modality bias embeddings
        self.modality_emb = nn.ParameterDict()
        if self.visual_proj is not None:
            self.modality_emb['vis'] = nn.Parameter(torch.zeros(self.d_model))
        if self.pose_proj is not None:
            self.modality_emb['pose'] = nn.Parameter(torch.zeros(self.d_model))
        if self.gaze_proj is not None:
            self.modality_emb['gaze'] = nn.Parameter(torch.zeros(self.d_model))
        if self.motion_proj is not None:
            self.modality_emb['motion'] = nn.Parameter(torch.zeros(self.d_model))
        if self.goal_proj is not None:
            self.modality_emb['goal'] = nn.Parameter(torch.zeros(self.d_model))

        # time embeddings
        self.time_emb = nn.Parameter(torch.randn(self.max_time_steps, self.d_model) * 0.02)

        # transformer encoder for self-attention over visual stream and optional post-conditioning mixing
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # cross-attention (visual queries -> modality keys/values)
        # nn.MultiheadAttention expects (L, N, E) so we will transpose when calling
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=nhead, dropout=dropout)
        self.cross_attn_ln = nn.LayerNorm(self.d_model)

        # gate network for adaptive goal conditioning: maps fused token -> scalar in [0,1]
        self.gate_net = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(8, self.d_model // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, self.d_model // 4), 1),
            nn.Sigmoid(),
        )

        # classification head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(16, self.d_model // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(16, self.d_model // 2), n_classes),
        )

        # optional future prediction head
        self.predict_future = int(predict_future_steps) > 0
        if self.predict_future:
            self.future_head = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(inplace=True),
                nn.Linear(self.d_model, int(predict_future_steps) * 2),
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for k, p in self.modality_emb.items():
            nn.init.normal_(p, std=0.02)
        # init linear layers if present
        for name in ('visual_proj', 'pose_proj', 'gaze_proj', 'motion_proj', 'goal_proj'):
            m = getattr(self, name)
            if m is not None:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, visual: torch.Tensor, head_pose: Optional[torch.Tensor] = None, motion: Optional[torch.Tensor] = None, gaze: Optional[torch.Tensor] = None, goal: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            visual: (B, T, V) or (B, T, H, W, C) if raw frames - we expect flattened or backbone embeddings.
            head_pose: (B, T, P)
            motion: (B, T, M)
            gaze: (B, T, G)
            goal: (B, T, 3) with [d, sin, cos]

        Returns:
            logits: (B, n_classes)
            future_pred: (B, predict_future_steps, 2) if enabled else None
        """
        B = visual.shape[0]
        T = visual.shape[1]

        # project visual per-frame features
        v = visual
        if v.dim() == 5:
            # flatten H,W,C
            v = v.view(B, T, -1)
        v_emb = self.visual_proj(v) if self.visual_proj is not None else v
        if 'vis' in self.modality_emb:
            v_emb = v_emb + self.modality_emb['vis']

        # 1) Self-attend over visual stream -> visual context
        visual_ctx = self.transformer(v_emb)  # (B, T, d)

        # 2) Cross-attend from visual queries to each available modality and sum contributions
        cross_sources = []
        if head_pose is not None and self.pose_proj is not None:
            p_emb = self.pose_proj(head_pose)
            p_emb = p_emb + self.modality_emb.get('pose', 0.0)
            cross_sources.append(p_emb)
        if motion is not None and self.motion_proj is not None:
            m_emb = self.motion_proj(motion)
            m_emb = m_emb + self.modality_emb.get('motion', 0.0)
            cross_sources.append(m_emb)
        if gaze is not None and self.gaze_proj is not None:
            g_emb = self.gaze_proj(gaze)
            g_emb = g_emb + self.modality_emb.get('gaze', 0.0)
            cross_sources.append(g_emb)

        fused = visual_ctx
        if len(cross_sources) > 0:
            # prepare query/key/value for MultiheadAttention: (L, N, E) where L=seq len, N=batch
            q = fused.transpose(0, 1)  # (T, B, d)
            attn_sum = None
            for src in cross_sources:
                k = src.transpose(0, 1)  # (T, B, d)
                v = k
                attn_out, _ = self.cross_attn(q, k, v)  # (T, B, d)
                attn_out = attn_out.transpose(0, 1)  # (B, T, d)
                attn_sum = attn_out if attn_sum is None else attn_sum + attn_out
            # residual and norm
            fused = fused + self.cross_attn_ln(attn_sum)

        # 3) Add time embeddings
        te = self.time_emb[:T, :].unsqueeze(0).to(fused.device)
        fused = fused + te

        # 4) Adaptive goal conditioning: mix fused with goal embedding per timestep using learned gate
        if goal is not None and self.goal_proj is not None:
            g_emb = self.goal_proj(goal)
            g_emb = g_emb + self.modality_emb.get('goal', 0.0)
            Uhat = self.gate_net(fused)  # (B, T, 1)
            fused = (1.0 - Uhat) * fused + Uhat * g_emb

        # 5) Optionally allow further temporal mixing
        x = self.transformer(fused)

        # pool and classify
        pooled = x.mean(dim=1)  # (B, d)
        pooled = pooled + self.cls_token.squeeze(1)
        logits = self.classifier(pooled)

        # optionally produce future prediction but return only logits for compatibility
        # with the training harness which expects a single tensor result.
        # If future prediction is needed externally, call the future_head on pooled features.
        return logits


def build_slim_model_from_config(cfg: dict) -> SlimFusionTransformer:
    return SlimFusionTransformer(
        visual_dim=cfg.get('visual_dim', 512),
        pose_dim=cfg.get('pose_dim', 3),
        gaze_dim=cfg.get('gaze_dim', 0),
        motion_dim=cfg.get('motion_dim', 2),
        goal_dim=cfg.get('goal_dim', 3),
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 3),
        dim_feedforward=cfg.get('dim_feedforward', 256),
        dropout=cfg.get('dropout', 0.1),
        n_classes=cfg.get('n_classes', 5),
        max_time_steps=cfg.get('max_time_steps', 64),
        predict_future_steps=cfg.get('predict_future_steps', 0),
    )


if __name__ == '__main__':
    # quick shape-test
    m = SlimFusionTransformer(visual_dim=256, pose_dim=3, gaze_dim=2, motion_dim=2, goal_dim=3, d_model=128, n_classes=4)
    B, T = 2, 8
    vis = torch.randn(B, T, 256)
    pose = torch.randn(B, T, 3)
    motion = torch.randn(B, T, 2)
    gaze = torch.randn(B, T, 2)
    goal = torch.randn(B, T, 3)
    logits = m(vis, head_pose=pose, motion=motion, gaze=gaze, goal=goal)
    print('logits', logits.shape)
