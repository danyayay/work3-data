"""Egocentric data fusion model.

This file implements a Transformer-based fusion network that consumes
time-series of visual features and optional head-pose and gaze inputs,
and predicts a future intention (classification) after a short horizon
(e.g. 1 second).

Design summary:
- Expect pre-extracted visual embeddings per frame (B, T, C_v).
- Optional head pose (B, T, C_p) and gaze (B, T, C_g).
- Project each modality to a shared d_model, add learned temporal
  positional embeddings and modality-type embeddings, concatenate tokens,
  run a stack of Transformer encoder layers to fuse information across
  time and modalities, then classifier head.

The model is intentionally modular so you can replace the visual
backbone with your preferred extractor or pass-in features from an
existing backbone in the repo.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EgocentricFusionTransformer(nn.Module):
	"""Transformer-based fusion model for egocentric time-series.

	Args:
		visual_dim: dimensionality of input visual features per timestep.
		pose_dim: dimensionality of head-pose input per timestep (0 to disable).
		gaze_dim: dimensionality of gaze input per timestep (0 to disable).
		d_model: embedding dimension inside the transformer.
		nhead: number of attention heads.
		num_layers: number of transformer encoder layers for fusion.
		dim_feedforward: MLP dimension inside transformer.
		dropout: dropout probability.
		n_classes: number of output intention classes.
		use_cls_token: whether to prepend a learnable CLS token for pooling.
	"""

	def __init__(
		self,
		visual_dim: int,
		pose_dim: int = 0,
		gaze_dim: int = 0,
		motion_dim: int = 0,
		goal_dim: int = 0,
		d_model: int = 256,
		nhead: int = 8,
		num_layers: int = 4,
		dim_feedforward: int = 512,
		dropout: float = 0.1,
		n_classes: int = 10,
		use_cls_token: bool = True,
		max_time_steps: int = 64,
	):
		super().__init__()

		self.visual_dim = visual_dim
		self.pose_dim = pose_dim
		self.gaze_dim = gaze_dim
		self.motion_dim = motion_dim
		self.goal_dim = goal_dim
		self.d_model = d_model
		self.use_cls_token = use_cls_token

		# modality projection layers
		self.visual_proj = nn.Linear(visual_dim, d_model) if visual_dim > 0 else None
		self.pose_proj = nn.Linear(pose_dim, d_model) if pose_dim > 0 else None
		self.gaze_proj = nn.Linear(gaze_dim, d_model) if gaze_dim > 0 else None
		self.motion_proj = nn.Linear(motion_dim, d_model) if motion_dim > 0 else None
		self.goal_proj = nn.Linear(goal_dim, d_model) if goal_dim > 0 else None

		# learned temporal positional embeddings (shared across modalities)
		self.max_time_steps = max_time_steps
		self.pos_emb = nn.Parameter(torch.zeros(1, max_time_steps, d_model))

		# modality type embeddings so transformer can tell modalities apart
		self.modality_emb = nn.Parameter(torch.randn(5, d_model) * 0.02)
		# order: visual=0, pose=1, gaze=2, motion=3, goal=4

		# optional CLS token for pooled representation
		if use_cls_token:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
		else:
			self.cls_token = None

		# Transformer encoder for fusion
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			activation="gelu",
			batch_first=True,
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		# classification head
		self.pool_norm = nn.LayerNorm(d_model)
		self.classifier = nn.Sequential(
			nn.Linear(d_model, d_model // 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_model // 2, n_classes)
		)

		self._reset_parameters()

	def _reset_parameters(self):
		# init weights similar to standard transformer practice
		if self.visual_proj is not None:
			nn.init.xavier_normal_(self.visual_proj.weight)
			nn.init.constant_(self.visual_proj.bias, 0.0)
		if self.pose_proj is not None:
			nn.init.xavier_normal_(self.pose_proj.weight)
			nn.init.constant_(self.pose_proj.bias, 0.0)
		if self.gaze_proj is not None:
			nn.init.xavier_normal_(self.gaze_proj.weight)
			nn.init.constant_(self.gaze_proj.bias, 0.0)
		if self.motion_proj is not None:
			nn.init.xavier_normal_(self.motion_proj.weight)
			nn.init.constant_(self.motion_proj.bias, 0.0)
		if self.goal_proj is not None:
			nn.init.xavier_normal_(self.goal_proj.weight)
			nn.init.constant_(self.goal_proj.bias, 0.0)
		nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
		nn.init.normal_(self.modality_emb, mean=0.0, std=0.02)
		if self.cls_token is not None:
			nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

	def forward(
		self,
		visual_feats: torch.Tensor,
		head_pose: Optional[torch.Tensor] = None,
		gaze: Optional[torch.Tensor] = None,
		motion: Optional[torch.Tensor] = None,
		goal: Optional[torch.Tensor] = None,
		time_mask: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""Forward pass.

		Args:
			visual_feats: (B, T, C_v) required.
			head_pose: (B, T, C_p) or None.
			gaze: (B, T, C_g) or None.
			motion: (B, T, C_m) or None. Past body-frame motion (e.g. pedestrian location) per timestep.
			goal: (B, T, C_g) or None. Per-timestep goal representation [d, sin(beta), cos(beta)].
			time_mask: optional boolean mask (B, T) where True indicates valid
				timesteps. If provided, invalid timesteps will be masked.

		Returns:
			logits: (B, n_classes)
		"""
		B, T, _ = visual_feats.shape

		if T > self.max_time_steps:
			raise ValueError(f"T={T} exceeds configured max_time_steps={self.max_time_steps}")

		tokens = []
		token_types = []

		# visual tokens
		v = self.visual_proj(visual_feats)  # B, T, d_model
		v = v + self.pos_emb[:, :T, :]
		tokens.append(v)
		token_types.append(torch.zeros(1, 1, dtype=torch.long, device=v.device))

		# pose tokens
		if (self.pose_proj is not None) and (head_pose is not None):
			p = self.pose_proj(head_pose)
			p = p + self.pos_emb[:, :T, :]
			tokens.append(p)
			token_types.append(torch.ones(1, 1, dtype=torch.long, device=p.device))

		# gaze tokens
		if (self.gaze_proj is not None) and (gaze is not None):
			g = self.gaze_proj(gaze)
			g = g + self.pos_emb[:, :T, :]
			tokens.append(g)
			token_types.append(2 * torch.ones(1, 1, dtype=torch.long, device=g.device))

		# motion tokens
		if (self.motion_proj is not None) and (motion is not None):
			m = self.motion_proj(motion)
			m = m + self.pos_emb[:, :T, :]
			tokens.append(m)
			token_types.append(3 * torch.ones(1, 1, dtype=torch.long, device=m.device))

		# goal tokens
		if (self.goal_proj is not None) and (goal is not None):
			g_goal = self.goal_proj(goal)
			g_goal = g_goal + self.pos_emb[:, :T, :]
			tokens.append(g_goal)
			token_types.append(4 * torch.ones(1, 1, dtype=torch.long, device=g_goal.device))

		# Concatenate tokens along token dimension: modality-wise blocks
		# tokens: list of (B, T, d_model) -> concat -> (B, M*T, d_model)
		x = torch.cat(tokens, dim=1)

		# Add modality embeddings to each block
		# Build a repeated type embedding per-token
		# Example: if tokens = [vis, pose, gaze] then types = [0]*T + [1]*T + [2]*T
		type_embeddings = []
		for i, block in enumerate(tokens):
			# block shape (B, T, d_model)
			t = self.modality_emb[i : i + 1, :].unsqueeze(1)  # 1,1,d
			t = t.expand(1, block.size(1), -1)  # 1,T,d
			type_embeddings.append(t)
		type_embeddings = torch.cat(type_embeddings, dim=1).expand(B, -1, -1)
		x = x + type_embeddings

		# Optionally prepend CLS token
		if self.cls_token is not None:
			cls = self.cls_token.expand(B, -1, -1)  # B,1,d
			x = torch.cat([cls, x], dim=1)

		# Apply attention mask if time_mask provided
		# Transformer expects src_key_padding_mask of shape (B, S) where True indicates padding.
		src_key_padding_mask = None
		if time_mask is not None:
			# time_mask: (B, T) -> need to expand to match concatenated tokens
			masks = []
			for block in tokens:
				masks.append(~time_mask)  # invert: True=pad
			pad_block = torch.cat(masks, dim=1)  # (B, M*T)
			if self.cls_token is not None:
				# cls token is never masked
				pad_block = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad_block.device), pad_block], dim=1)
			src_key_padding_mask = pad_block

		# Transformer encoder
		x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

		# Pooling: use CLS token if present, else mean pool visual tokens
		if self.cls_token is not None:
			pooled = x[:, 0, :]
		else:
			# assume visual block is first block of length T
			pooled = x[:, :T, :].mean(dim=1)

		pooled = self.pool_norm(pooled)
		logits = self.classifier(pooled)

		return logits


class SimpleCNNBackbone(nn.Module):
	"""Small CNN to extract per-frame embeddings from raw frames.

	Expects input frames as float tensors in range [0,1] with shape
	(B, T, H, W, 3). Outputs per-frame embeddings (B, T, out_dim).
	"""

	def __init__(self, in_ch: int = 3, out_dim: int = 256, hidden: int = 64):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, hidden, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(hidden),
			nn.GELU(),
			nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(hidden * 2),
			nn.GELU(),
			nn.Conv2d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(hidden * 4),
			nn.GELU(),
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(nn.Linear(hidden * 4, out_dim), nn.GELU())

	def forward(self, frames: torch.Tensor) -> torch.Tensor:
		# frames: (B, T, H, W, 3) -> process as (B*T, C, H, W)
		B, T, H, W, C = frames.shape
		x = frames.permute(0, 1, 4, 2, 3).contiguous()  # B,T,C,H,W
		x = x.view(B * T, C, H, W)
		x = self.conv(x)
		x = self.pool(x).view(B * T, -1)
		x = self.fc(x)
		x = x.view(B, T, -1)
		return x


class CLIPBackbone(nn.Module):
	"""Wrapper around OpenAI CLIP visual encoder.

	Expects input frames as float tensors in range [0,1] with shape
	(B, T, H, W, 3). Returns per-frame embeddings (B, T, D).
	"""

	def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None, freeze: bool = False):
		super().__init__()
		try:
			import clip
		except Exception as e:
			raise ImportError("clip package is required for CLIPBackbone. Install via `pip install git+https://github.com/openai/CLIP.git` or provide a different encoder.") from e

		dev = device or torch.device("cpu")
		self.model, _ = clip.load(model_name, device=dev, jit=False)
		# set train/eval according to freeze
		if freeze:
			for p in self.model.parameters():
				p.requires_grad = False

	@property
	def out_dim(self) -> int:
		# CLIP visual dimensionality
		if hasattr(self.model.visual, "output_dim"):
			return self.model.visual.output_dim
		if hasattr(self.model, "output_dim"):
			return self.model.output_dim
		# fallback
		return 512

	def forward(self, frames: torch.Tensor) -> torch.Tensor:
		# frames: (B, T, H, W, 3)
		B, T, H, W, C = frames.shape
		x = frames.permute(0, 1, 4, 2, 3).contiguous().view(B * T, C, H, W)
		x = x.to(next(self.model.parameters()).device)
		# normalize with CLIP stats
		mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
		std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
		x = (x - mean) / std

		feats = self.model.encode_image(x)
		feats = feats.view(B, T, -1)
		return feats


if __name__ == "__main__":
	# quick shape check
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	B, T = 2, 8
	visual_dim = 512
	pose_dim = 6
	gaze_dim = 2
	n_classes = 5

	model = EgocentricFusionTransformer(
		visual_dim=visual_dim,
		pose_dim=pose_dim,
		gaze_dim=gaze_dim,
		motion_dim=2,
		d_model=128,
		nhead=8,
		num_layers=2,
		n_classes=n_classes,
		max_time_steps=32,
	).to(device)

	vis = torch.randn(B, T, visual_dim, device=device)
	pose = torch.randn(B, T, pose_dim, device=device)
	gaze = torch.randn(B, T, gaze_dim, device=device)
	motion = torch.randn(B, T, 2, device=device)
	goal = torch.randn(B, T, 3, device=device)

	logits = model(vis, head_pose=pose, gaze=gaze, motion=motion, goal=goal)
	print("logits.shape:", logits.shape)  # expected (B, n_classes)