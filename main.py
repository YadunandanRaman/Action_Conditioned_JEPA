"""

UNIFIED AGENT ARCHITECTURE FOR MINIGRID

Complete implementation of:
- CNN-based State Encoder for visual grid observations
- Transformer World Model for dynamics reasoning
- Curiosity-driven exploration (ICM + RND)
- PPO Policy with transformer backbone
- Full training pipeline for MiniGrid environments

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
import random
from collections import deque, defaultdict
import os
import time
from pathlib import Path
import logging
import gymnasium as gym

# MiniGrid imports
try:
    import minigrid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, FullyObsWrapper
    MINIGRID_AVAILABLE = True
except ImportError:
    MINIGRID_AVAILABLE = False
    print("WARNING: MiniGrid not installed. Run: pip install minigrid")

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# PART 1: CONFIGURATION

@dataclass
class CNNEncoderConfig:
    """Configuration for CNN-based state encoder."""
    input_channels: int = 3  # RGB
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 1])
    use_batch_norm: bool = True
    use_residual: bool = True
    dropout: float = 0.1


@dataclass
class TransformerConfig:
    """Configuration for transformer-based components."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    use_rotary_embeddings: bool = True
    layer_norm_eps: float = 1e-6
    pre_norm: bool = True


@dataclass
class PolicyConfig:
    """Configuration for the policy network."""
    hidden_dim: int = 256
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class CuriosityConfig:
    """Configuration for curiosity-driven exploration."""
    hidden_dim: int = 256
    feature_dim: int = 128
    forward_loss_coef: float = 0.2
    inverse_loss_coef: float = 0.8
    intrinsic_reward_scale: float = 0.1
    normalize_rewards: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    warmup_steps: int = 1000
    max_steps: int = 1000000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 256
    mini_batch_size: int = 64
    ppo_epochs: int = 4
    world_model_updates_per_step: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: bool = False 
    n_envs: int = 8  
    rollout_length: int = 128


@dataclass
class BufferConfig:
    """Configuration for experience buffer."""
    capacity: int = 100000
    prioritized: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_frames: int = 100000


@dataclass
class AgentConfig:
    """Complete agent configuration."""
    # Observation dimensions (will be set based on environment)
    obs_shape: Tuple[int, int, int] = (7, 7, 3)  # MiniGrid default partial obs
    latent_dim: int = 256
    action_dim: int = 7  # MiniGrid action space
    
    # Sub-configurations
    cnn: CNNEncoderConfig = field(default_factory=CNNEncoderConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Environment settings
    env_name: str = "MiniGrid-Empty-8x8-v0"
    use_full_obs: bool = False  # Use partial observation (agent's view)
    use_rgb: bool = True  # Use RGB rendering


# PART 2: MATHEMATICAL UTILITIES

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def initialize_weights(module: nn.Module, std: float = 0.02):
    """Initialize weights with truncated normal distribution."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=std)
    elif isinstance(module, (nn.LayerNorm, RMSNorm, nn.BatchNorm2d)):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)


# PART 3: CNN STATE ENCODER

class ResidualBlock(nn.Module):
    """Residual block for CNN encoder."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=not use_batch_norm)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=not use_batch_norm)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                               stride=stride, bias=False)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        out = out + identity
        out = F.relu(out)
        
        return out


class CNNStateEncoder(nn.Module):
    """
    CNN-based State Encoder for visual observations.
    
    This is the PERCEPTION component for grid-world environments.
    Processes visual input into latent representations.
    """
    def __init__(self, config: CNNEncoderConfig, obs_shape: Tuple[int, int, int], 
                 latent_dim: int):
        super().__init__()
        
        self.config = config
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        
        # Input shape: (H, W, C) -> need (C, H, W)
        h, w, c = obs_shape
        
        # Build convolutional layers
        layers = []
        in_channels = c
        
        for i, (out_channels, kernel_size, stride) in enumerate(zip(
            config.hidden_channels, config.kernel_sizes, config.strides
        )):
            if config.use_residual and i > 0:
                layers.append(ResidualBlock(
                    in_channels, out_channels, stride, config.use_batch_norm
                ))
            else:
                layers.append(nn.Conv2d(
                    in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=kernel_size // 2
                ))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).shape[1]
        
        # Projection to latent space
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_size, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Softplus()
        )
        
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observation to latent representation.
        
        Args:
            obs: [batch, H, W, C] or [batch, C, H, W]
        
        Returns:
            latent: [batch, latent_dim]
        """
        # Handle channel dimension
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        # Convert from (B, H, W, C) to (B, C, H, W) if needed
        if obs.shape[-1] == self.obs_shape[-1] and obs.shape[1] != self.obs_shape[-1]:
            obs = obs.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] if integer input
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        elif obs.max() > 1.0:
            obs = obs.float() / 255.0
        
        # Convolutional encoding
        features = self.conv_layers(obs)
        features = features.reshape(features.shape[0], -1)
        
        # Project to latent space
        latent = self.projection(features)
        
        return latent
    
    def encode_with_uncertainty(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode with uncertainty estimation."""
        latent = self.forward(obs)
        uncertainty = self.uncertainty_head(latent).squeeze(-1)
        return latent, uncertainty


class SequentialCNNEncoder(nn.Module):
    """
    Encodes sequences of visual observations using CNN + Transformer.
    
    For temporal reasoning over observation history.
    """
    def __init__(self, config: CNNEncoderConfig, transformer_config: TransformerConfig,
                 obs_shape: Tuple[int, int, int], latent_dim: int):
        super().__init__()
        
        # Frame encoder
        self.frame_encoder = CNNStateEncoder(config, obs_shape, latent_dim)
        
        # Temporal transformer
        self.pos_embedding = nn.Parameter(
            torch.randn(1, transformer_config.max_seq_len, latent_dim) * 0.02
        )
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                d_model=latent_dim,
                n_heads=transformer_config.n_heads,
                d_ff=transformer_config.d_ff,
                dropout=transformer_config.dropout,
                use_rotary=transformer_config.use_rotary_embeddings,
                max_seq_len=transformer_config.max_seq_len,
                is_causal=True
            )
            for _ in range(transformer_config.n_layers // 2)  # Fewer layers for sequence
        ])
        
        self.norm = RMSNorm(latent_dim)
    
    def forward(self, obs_sequence: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sequence of observations.
        
        Args:
            obs_sequence: [batch, seq_len, H, W, C]
            mask: [batch, seq_len] - True for padded positions
        
        Returns:
            encoded: [batch, seq_len, latent_dim]
        """
        batch_size, seq_len = obs_sequence.shape[:2]
        
        # Encode each frame
        obs_flat = obs_sequence.view(-1, *obs_sequence.shape[2:])
        latents_flat = self.frame_encoder(obs_flat)
        latents = latents_flat.view(batch_size, seq_len, -1)
        
        # Add positional encoding
        latents = latents + self.pos_embedding[:, :seq_len, :]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            latents, _ = layer(latents, key_padding_mask=mask)
        
        latents = self.norm(latents)
        
        return latents


# ============================================================================
# PART 4: TRANSFORMER COMPONENTS
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional rotary embeddings."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_rotary: bool = True, max_seq_len: int = 2048,
                 is_causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = is_causal
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = query.shape
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        if self.use_rotary:
            cos, sin = self.rotary_emb(q, seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Feed-Forward Network with SwiGLU activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff * 2, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = SwiGLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-normalization."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 use_rotary: bool = True, max_seq_len: int = 2048, is_causal: bool = False):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, use_rotary, 
                                        max_seq_len, is_causal)
        self.norm2 = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.attn(x, x, x, attention_mask, key_padding_mask)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x) + residual
        
        return x, attn_weights


# PART 5: ACTION ENCODER/DECODER

class ActionEncoder(nn.Module):
    """Encodes discrete actions into latent space."""
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(action_dim, latent_dim)
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.embedding(action)


class ActionDecoder(nn.Module):
    """Decodes latent representations to action logits."""
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


# PART 6: TRANSFORMER WORLD MODEL

class TransformerWorldModel(nn.Module):
    """
    THE CORE REASONING ENGINE FOR MINIGRID
    
    Predicts state dynamics in latent space:
    - Given: current latent state + action
    - Predicts: next latent state, reward, done, uncertainty
    """
    def __init__(self, config: TransformerConfig, latent_dim: int, action_dim: int):
        super().__init__()
        
        self.config = config
        self.latent_dim = latent_dim
        self.d_model = config.d_model
        
        # Project state and action to model dimension
        self.state_proj = nn.Linear(latent_dim, config.d_model)
        self.action_encoder = ActionEncoder(action_dim, config.d_model)
        
        # Type embeddings
        self.type_embedding = nn.Embedding(3, config.d_model)  # state, action, prediction
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                use_rotary=config.use_rotary_embeddings,
                max_seq_len=config.max_seq_len,
                is_causal=False
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.d_model)
        
        # Output heads
        self.state_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, latent_dim)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Softplus()
        )
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1)
        )
        
        self.done_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, 1)
        )
        
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, latent_state: torch.Tensor, action: torch.Tensor,
                return_uncertainty: bool = False, return_attention: bool = False
                ) -> Dict[str, torch.Tensor]:
        batch_size = latent_state.shape[0]
        device = latent_state.device
        
        # Project state and action
        state_emb = self.state_proj(latent_state)
        action_emb = self.action_encoder(action)
        
        # Add type embeddings
        state_type = self.type_embedding(torch.zeros(batch_size, dtype=torch.long, device=device))
        action_type = self.type_embedding(torch.ones(batch_size, dtype=torch.long, device=device))
        
        state_emb = state_emb + state_type
        action_emb = action_emb + action_type
        
        # Create sequence
        sequence = torch.stack([state_emb, action_emb], dim=1)
        
        # Apply transformer layers
        attention_weights_all = []
        for layer in self.layers:
            sequence, attn_weights = layer(sequence)
            attention_weights_all.append(attn_weights)
        
        sequence = self.norm(sequence)
        final_repr = sequence[:, -1, :]
        
        # Predictions
        next_state = self.state_predictor(final_repr)
        done_logits = self.done_predictor(final_repr).squeeze(-1)
        
        output = {
            'next_state': next_state,
            'predicted_reward': self.reward_predictor(final_repr).squeeze(-1),
            'predicted_done_logits': done_logits,
            'predicted_done': torch.sigmoid(done_logits)
        }
        
        if return_uncertainty:
            output['uncertainty'] = self.uncertainty_head(final_repr).squeeze(-1)
        
        if return_attention:
            output['attention'] = torch.stack(attention_weights_all, dim=1)
        
        return output
    
    def imagine_trajectory(self, initial_latent: torch.Tensor, 
                           action_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Simulate future trajectory in imagination."""
        batch_size = initial_latent.shape[0]
        
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0).expand(batch_size, -1)
        
        horizon = action_sequence.shape[1]
        
        trajectory = [initial_latent]
        rewards = []
        dones = []
        uncertainties = []
        
        current_state = initial_latent
        
        for t in range(horizon):
            action = action_sequence[:, t]
            output = self.forward(current_state, action, return_uncertainty=True)
            
            trajectory.append(output['next_state'])
            rewards.append(output['predicted_reward'])
            dones.append(output['predicted_done'])
            uncertainties.append(output['uncertainty'])
            
            current_state = output['next_state']
        
        return {
            'trajectory': torch.stack(trajectory, dim=1),
            'rewards': torch.stack(rewards, dim=1),
            'dones': torch.stack(dones, dim=1),
            'uncertainties': torch.stack(uncertainties, dim=1)
        }
    
    def compute_prediction_loss(self, latent_states: torch.Tensor, actions: torch.Tensor,
                                 next_latent_states: torch.Tensor,
                                 rewards: Optional[torch.Tensor] = None,
                                 dones: Optional[torch.Tensor] = None
                                 ) -> Dict[str, torch.Tensor]:
        output = self.forward(latent_states, actions, return_uncertainty=True)
        
        state_loss = F.mse_loss(output['next_state'], next_latent_states)
        losses = {'state_loss': state_loss}
        
        if rewards is not None:
            reward_loss = F.mse_loss(output['predicted_reward'], rewards)
            losses['reward_loss'] = reward_loss
        
        if dones is not None:
            done_loss = F.binary_cross_entropy_with_logits(
                output['predicted_done_logits'], dones.float()
            )
            losses['done_loss'] = done_loss
        
        total_loss = state_loss
        if rewards is not None:
            total_loss = total_loss + 0.5 * losses['reward_loss']
        if dones is not None:
            total_loss = total_loss + 0.1 * losses['done_loss']
        
        losses['total_loss'] = total_loss
        return losses


# PART 7: INTRINSIC CURIOSITY MODULE

class IntrinsicCuriosityModule(nn.Module):
    """
    ICM for curiosity-driven exploration.
    
    Forward model: predicts next state features from current state + action
    Inverse model: predicts action from current and next state features
    """
    def __init__(self, config: CuriosityConfig, latent_dim: int, action_dim: int):
        super().__init__()
        
        self.config = config
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim)
        )
        
        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(config.feature_dim + action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim)
        )
        
        # Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, action_dim)
        )
        
        # Running statistics
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_var', torch.ones(1))
        self.register_buffer('reward_count', torch.zeros(1))
        
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, latent_state: torch.Tensor, action: torch.Tensor,
                next_latent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        state_feat = self.feature_net(latent_state)
        next_state_feat = self.feature_net(next_latent_state)
        
        action_onehot = F.one_hot(action, self.action_dim).float()
        
        forward_input = torch.cat([state_feat, action_onehot], dim=-1)
        predicted_next_feat = self.forward_model(forward_input)
        
        forward_error = F.mse_loss(predicted_next_feat, next_state_feat.detach(), reduction='none')
        forward_error_per_sample = forward_error.mean(dim=-1)
        forward_loss = forward_error_per_sample.mean()
        
        inverse_input = torch.cat([state_feat, next_state_feat], dim=-1)
        predicted_action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(predicted_action_logits, action)
        
        intrinsic_reward = forward_error_per_sample.detach()
        
        if self.config.normalize_rewards and intrinsic_reward.numel() > 1:
            intrinsic_reward = self._normalize_reward(intrinsic_reward)
        
        intrinsic_reward = intrinsic_reward * self.config.intrinsic_reward_scale
        
        return {
            'intrinsic_reward': intrinsic_reward,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss,
            'total_loss': (
                self.config.forward_loss_coef * forward_loss +
                self.config.inverse_loss_coef * inverse_loss
            )
        }
    
    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        batch_mean = reward.mean()
        batch_var = reward.var()
        batch_count = reward.numel()
        
        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count
        
        self.reward_mean = self.reward_mean + delta * batch_count / total_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / total_count
        self.reward_var = M2 / total_count
        self.reward_count = total_count
        
        return (reward - self.reward_mean) / (torch.sqrt(self.reward_var) + 1e-8)


class RandomNetworkDistillation(nn.Module):
    """RND for novelty-based exploration."""
    def __init__(self, latent_dim: int, feature_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        self.target_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.predictor_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        self.register_buffer('obs_mean', torch.zeros(latent_dim))
        self.register_buffer('obs_var', torch.ones(latent_dim))
        self.register_buffer('reward_mean', torch.zeros(1))
        self.register_buffer('reward_var', torch.ones(1))
        
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, latent_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized_state = (latent_state - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-8)
        
        with torch.no_grad():
            target_features = self.target_net(normalized_state)
        
        predicted_features = self.predictor_net(normalized_state)
        error = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=-1)
        normalized_reward = (error - self.reward_mean) / (torch.sqrt(self.reward_var) + 1e-8)
        
        return {
            'intrinsic_reward': normalized_reward.detach(),
            'loss': error.mean()
        }


# PART 8: POLICY NETWORK

class PolicyNetwork(nn.Module):
    """
    Transformer-based Policy Network for MiniGrid.
    
    Processes latent states and outputs action distributions + value estimates.
    """
    def __init__(self, config: PolicyConfig, latent_dim: int, action_dim: int):
        super().__init__()
        
        self.config = config
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        self.input_proj = nn.Linear(latent_dim, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.hidden_dim,
                n_heads=config.n_heads,
                d_ff=config.hidden_dim * 4,
                dropout=config.dropout,
                use_rotary=True,
                max_seq_len=512
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_dim)
        
        self.action_head = ActionDecoder(config.hidden_dim, action_dim, config.hidden_dim)
        
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.apply(lambda m: initialize_weights(m, std=0.02))
    
    def forward(self, latent_state: torch.Tensor, 
                return_value: bool = True) -> Dict[str, torch.Tensor]:
        if latent_state.dim() == 2:
            latent_state = latent_state.unsqueeze(1)
        
        x = self.input_proj(latent_state)
        
        for layer in self.layers:
            x, _ = layer(x)
        
        x = self.norm(x)
        features = x[:, -1, :]
        
        action_logits = self.action_head(features)
        output = {'action_logits': action_logits}
        
        if return_value:
            output['value'] = self.value_head(features).squeeze(-1)
        
        return output
    
    def get_action(self, latent_state: torch.Tensor, deterministic: bool = False,
                   temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        output = self.forward(latent_state, return_value=True)
        
        logits = output['action_logits'] / temperature
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        
        return {
            'action': action,
            'log_prob': log_prob,
            'value': output['value'],
            'entropy': entropy
        }
    
    def evaluate_actions(self, latent_states: torch.Tensor, 
                         actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.forward(latent_states, return_value=True)
        
        logits = output['action_logits']
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        action_log_prob = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return {
            'log_prob': action_log_prob,
            'value': output['value'],
            'entropy': entropy
        }


# PART 9: EXPERIENCE BUFFERS

@dataclass
class Experience:
    """Single experience tuple."""
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    log_prob: float = 0.0
    value: float = 0.0
    intrinsic_reward: float = 0.0


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer."""
    def __init__(self, config: BufferConfig):
        self.capacity = config.capacity
        self.alpha = config.alpha
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.beta_frames = config.beta_frames
        
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.frame = 0
        self.max_priority = 1.0
    
    def add(self, experience: Experience, priority: float = None):
        if priority is None:
            priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        self.frame += 1
        
        beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * self.frame / self.beta_frames
        )
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """Buffer for on-policy PPO training."""
    def __init__(self, buffer_size: int, obs_shape: Tuple, gamma: float = 0.99, 
                 gae_lambda: float = 0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs_shape = obs_shape
        
        self.reset()
    
    def reset(self):
        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.intrinsic_rewards = np.zeros(self.buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False
    
    def add(self, obs: np.ndarray, action: int, reward: float, done: bool,
            log_prob: float, value: float, intrinsic_reward: float = 0.0):
        if self.ptr >= self.buffer_size:
            return
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.intrinsic_rewards[self.ptr] = intrinsic_reward
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        intrinsic = self.intrinsic_rewards[path_slice]
        total_rewards = rewards + intrinsic
        
        values = np.append(self.values[path_slice], last_value)
        dones = self.dones[path_slice]
        
        deltas = total_rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        
        advantages = np.zeros_like(deltas)
        last_gae = 0
        for t in reversed(range(len(deltas))):
            last_gae = deltas[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values[:-1]
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, torch.Tensor]:
        size = self.ptr
        
        advantages = self.advantages[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'observations': torch.FloatTensor(self.observations[:size]),
            'actions': torch.LongTensor(self.actions[:size]),
            'log_probs': torch.FloatTensor(self.log_probs[:size]),
            'values': torch.FloatTensor(self.values[:size]),
            'advantages': torch.FloatTensor(advantages),
            'returns': torch.FloatTensor(self.returns[:size])
        }


# PART 10: MINIGRID ENVIRONMENT WRAPPER

class MiniGridWrapper:
    """
    Wrapper for MiniGrid environments.
    
    Handles observation preprocessing and provides a clean interface.
    """
    def __init__(self, env_name: str, use_full_obs: bool = False, 
                 use_rgb: bool = True, max_steps: int = None):
        self.env_name = env_name
        self.use_full_obs = use_full_obs
        self.use_rgb = use_rgb
        
        # Create environment
        self.env = gym.make(env_name, render_mode='rgb_array')
        
        # Apply wrappers
        if use_full_obs:
            self.env = FullyObsWrapper(self.env)
        
        if use_rgb:
            self.env = RGBImgObsWrapper(self.env)
        else:
            self.env = ImgObsWrapper(self.env)
        
        if max_steps is not None:
            self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=max_steps)
        
        # Get observation and action spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Determine observation shape
        if isinstance(self.observation_space, gym.spaces.Box):
            self.obs_shape = self.observation_space.shape
        else:
            self.obs_shape = self.observation_space['image'].shape
        
        self.action_dim = self.action_space.n
        
        logger.info(f"MiniGrid Environment: {env_name}")
        logger.info(f"  Observation shape: {self.obs_shape}")
        logger.info(f"  Action space: {self.action_dim}")
    
    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        return self._process_obs(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._process_obs(obs), reward, done, truncated, info
    
    def _process_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs['image']
        return obs.astype(np.float32)
    
    def render(self) -> np.ndarray:
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def get_obs_shape(self) -> Tuple[int, int, int]:
        return self.obs_shape
    
    def get_action_dim(self) -> int:
        return self.action_dim


class VectorizedMiniGridEnv:
    """Vectorized MiniGrid environments for parallel training."""
    def __init__(self, env_name: str, n_envs: int, use_full_obs: bool = False,
                 use_rgb: bool = True, seed: int = 42):
        self.n_envs = n_envs
        self.envs = [
            MiniGridWrapper(env_name, use_full_obs, use_rgb)
            for _ in range(n_envs)
        ]
        
        # Seed environments
        for i, env in enumerate(self.envs):
            env.env.reset(seed=seed + i)
        
        self.obs_shape = self.envs[0].get_obs_shape()
        self.action_dim = self.envs[0].get_action_dim()
    
    def reset(self) -> np.ndarray:
        observations = np.stack([env.reset() for env in self.envs])
        return observations
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        next_obs = np.stack([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[4] for r in results]
        
        # Auto-reset done environments
        for i, done in enumerate(dones):
            if done:
                next_obs[i] = self.envs[i].reset()
        
        return next_obs, rewards, dones, infos
    
    def close(self):
        for env in self.envs:
            env.close()


# PART 11: EPISODIC MEMORY

class EpisodicMemory:
    """Long-term episodic memory for the agent."""
    def __init__(self, capacity: int, embedding_dim: int):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        
        self.embeddings = np.zeros((capacity, embedding_dim), dtype=np.float32)
        self.metadata = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def add(self, embedding: np.ndarray, metadata: Dict = None, priority: float = 1.0):
        self.embeddings[self.position] = embedding
        self.metadata[self.position] = metadata or {}
        self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def query(self, query_embedding: np.ndarray, k: int = 5
              ) -> List[Tuple[np.ndarray, Dict, float]]:
        if self.size == 0:
            return []
        
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        memory_norms = self.embeddings[:self.size] / (
            np.linalg.norm(self.embeddings[:self.size], axis=1, keepdims=True) + 1e-8
        )
        
        similarities = memory_norms @ query_norm
        weighted_similarities = similarities * self.priorities[:self.size]
        
        k = min(k, self.size)
        top_indices = np.argsort(weighted_similarities)[-k:][::-1]
        
        return [
            (self.embeddings[idx].copy(), self.metadata[idx], similarities[idx])
            for idx in top_indices
        ]


# PART 12: UNIFIED AGENT FOR MINIGRID

class UnifiedMiniGridAgent:
    """
    THE COMPLETE UNIFIED AGENT FOR MINIGRID
    
    Combines:
    - CNN State Encoder (Visual Perception)
    - Transformer World Model (Reasoning)
    - Policy Network (Agency)
    - Curiosity Module (Intrinsic Motivation)
    """
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        self._build_networks()
        self._build_optimizers()
        self._build_buffers()
        
        self.global_step = 0
        self.episodes_completed = 0
        self.best_reward = float('-inf')
        self.metrics = defaultdict(list)
        
        logger.info(f"Initialized UnifiedMiniGridAgent on {self.device}")
        logger.info(f"Total parameters: {self._count_parameters():,}")
    
    def _build_networks(self):
        config = self.config
        
        # CNN State Encoder
        self.encoder = CNNStateEncoder(
            config=config.cnn,
            obs_shape=config.obs_shape,
            latent_dim=config.latent_dim
        ).to(self.device)
        
        # Sequential Encoder
        self.seq_encoder = SequentialCNNEncoder(
            config=config.cnn,
            transformer_config=config.transformer,
            obs_shape=config.obs_shape,
            latent_dim=config.latent_dim
        ).to(self.device)
        
        # World Model
        self.world_model = TransformerWorldModel(
            config=config.transformer,
            latent_dim=config.latent_dim,
            action_dim=config.action_dim
        ).to(self.device)
        
        # Policy Network
        self.policy = PolicyNetwork(
            config=config.policy,
            latent_dim=config.latent_dim,
            action_dim=config.action_dim
        ).to(self.device)
        
        # Curiosity Modules
        self.curiosity = IntrinsicCuriosityModule(
            config=config.curiosity,
            latent_dim=config.latent_dim,
            action_dim=config.action_dim
        ).to(self.device)
        
        self.rnd = RandomNetworkDistillation(
            latent_dim=config.latent_dim,
            feature_dim=config.curiosity.feature_dim,
            hidden_dim=config.curiosity.hidden_dim
        ).to(self.device)
        
        # Target encoder
        self.target_encoder = CNNStateEncoder(
            config=config.cnn,
            obs_shape=config.obs_shape,
            latent_dim=config.latent_dim
        ).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    def _build_optimizers(self):
        config = self.config.training
        
        # World Model optimizer
        world_model_params = (
            list(self.encoder.parameters()) +
            list(self.world_model.parameters())
        )
        
        self.world_model_optimizer = torch.optim.AdamW(
            world_model_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
        
        # Policy optimizer
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate * 0.5,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
        
        # Curiosity optimizer
        curiosity_params = (
            list(self.curiosity.parameters()) +
            list(self.rnd.predictor_net.parameters())
        )
        
        self.curiosity_optimizer = torch.optim.AdamW(
            curiosity_params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay
        )
        
        # Schedulers
        self.world_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.world_model_optimizer, T_0=10000, T_mult=2,
            eta_min=config.learning_rate * 0.01
        )
        
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.policy_optimizer, T_0=10000, T_mult=2,
            eta_min=config.learning_rate * 0.01
        )
    
    def _build_buffers(self):
        config = self.config
        
        self.replay_buffer = PrioritizedReplayBuffer(config.buffer)
        
        self.rollout_buffer = RolloutBuffer(
            buffer_size=config.training.rollout_length * config.training.n_envs,
            obs_shape=config.obs_shape,
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda
        )
        
        self.episodic_memory = EpisodicMemory(
            capacity=10000,
            embedding_dim=config.latent_dim
        )
    
    def _count_parameters(self) -> int:
        total = 0
        for model in [self.encoder, self.world_model, self.policy, 
                      self.curiosity, self.seq_encoder]:
            total += sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total
    
    def encode(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Encode observation to latent representation."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        
        return self.encoder(obs)
    
    @torch.no_grad()
    def act(self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = False,
            temperature: float = 1.0) -> Dict[str, Any]:
        """Select action based on current observation."""
        self.policy.eval()
        
        latent = self.encode(obs)
        action_output = self.policy.get_action(latent, deterministic, temperature)
        
        if obs.shape[0] == 1 or (isinstance(obs, np.ndarray) and len(obs.shape) == 3):
            return {
                'action': action_output['action'].cpu().numpy()[0],
                'log_prob': action_output['log_prob'].cpu().numpy()[0],
                'value': action_output['value'].cpu().numpy()[0],
                'entropy': action_output['entropy'].cpu().numpy()[0],
                'latent': latent.cpu().numpy()[0]
            }
        else:
            return {
                'action': action_output['action'].cpu().numpy(),
                'log_prob': action_output['log_prob'].cpu().numpy(),
                'value': action_output['value'].cpu().numpy(),
                'entropy': action_output['entropy'].cpu().numpy(),
                'latent': latent.cpu().numpy()
            }
    
    @torch.no_grad()
    def predict_next_state(self, obs: Union[np.ndarray, torch.Tensor],
                           action: Union[int, np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """Predict next state given current state and action."""
        self.world_model.eval()
        
        latent = self.encode(obs)
        
        if isinstance(action, (int, np.integer)):
            action = torch.LongTensor([action]).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.LongTensor(action).to(self.device)
        
        output = self.world_model(latent, action, return_uncertainty=True)
        
        return {
            'next_latent': output['next_state'].cpu().numpy(),
            'predicted_reward': output['predicted_reward'].cpu().numpy(),
            'predicted_done': output['predicted_done'].cpu().numpy(),
            'uncertainty': output['uncertainty'].cpu().numpy()
        }
    
    @torch.no_grad()
    def imagine_trajectory(self, obs: Union[np.ndarray, torch.Tensor],
                           action_sequence: Union[List[int], np.ndarray]
                           ) -> Dict[str, np.ndarray]:
        """Simulate future trajectory in imagination."""
        self.world_model.eval()
        
        latent = self.encode(obs)
        
        if isinstance(action_sequence, list):
            action_sequence = torch.LongTensor(action_sequence).to(self.device)
        elif isinstance(action_sequence, np.ndarray):
            action_sequence = torch.LongTensor(action_sequence).to(self.device)
        
        output = self.world_model.imagine_trajectory(latent, action_sequence)
        
        return {
            'trajectory': output['trajectory'].cpu().numpy(),
            'rewards': output['rewards'].cpu().numpy(),
            'dones': output['dones'].cpu().numpy(),
            'uncertainties': output['uncertainties'].cpu().numpy()
        }
    
    @torch.no_grad()
    def compute_intrinsic_reward(self, obs: np.ndarray, action: int,
                                  next_obs: np.ndarray) -> float:
        """Compute curiosity-based intrinsic reward."""
        latent = self.encode(obs)
        next_latent = self.encode(next_obs)
        
        action_t = torch.LongTensor([action]).to(self.device)
        
        icm_output = self.curiosity(latent, action_t, next_latent)
        icm_reward = icm_output['intrinsic_reward'].cpu().numpy()[0]
        
        rnd_output = self.rnd(next_latent)
        rnd_reward = rnd_output['intrinsic_reward'].cpu().numpy()[0]
        
        return 0.5 * icm_reward + 0.5 * rnd_reward
    
    def store_experience(self, obs: np.ndarray, action: int, reward: float,
                         next_obs: np.ndarray, done: bool, log_prob: float = 0.0,
                         value: float = 0.0, intrinsic_reward: float = 0.0):
        """Store experience in buffers."""
        with torch.no_grad():
            latent = self.encode(obs)
            next_latent = self.encode(next_obs)
            
            action_t = torch.LongTensor([action]).to(self.device)
            wm_output = self.world_model(latent, action_t, return_uncertainty=True)
            
            pred_error = F.mse_loss(wm_output['next_state'], next_latent).item()
            priority = pred_error + abs(intrinsic_reward) + 0.1
        
        experience = Experience(
            obs=obs, action=action, reward=reward, next_obs=next_obs,
            done=done, log_prob=log_prob, value=value, intrinsic_reward=intrinsic_reward
        )
        
        self.replay_buffer.add(experience, priority)
        
        self.rollout_buffer.add(
            obs=obs, action=action, reward=reward, done=done,
            log_prob=log_prob, value=value, intrinsic_reward=intrinsic_reward
        )
        
        self.episodic_memory.add(
            embedding=latent.cpu().numpy()[0],
            metadata={'reward': reward, 'action': action}
        )
    
    def train_world_model(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Train the world model on prediction error."""
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return {}
        
        self.encoder.train()
        self.world_model.train()
        
        experiences, weights, indices = self.replay_buffer.sample(batch_size)
        
        obs = torch.FloatTensor(np.array([e.obs for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        next_obs = torch.FloatTensor(np.array([e.next_obs for e in experiences])).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Encode
        latents = self.encoder(obs)
        
        with torch.no_grad():
            target_next_latents = self.target_encoder(next_obs)
        
        # World model prediction
        losses = self.world_model.compute_prediction_loss(
            latent_states=latents,
            actions=actions,
            next_latent_states=target_next_latents,
            rewards=rewards,
            dones=dones
        )
        
        weighted_loss = (losses['total_loss'] * weights).mean()
        
        self.world_model_optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.world_model.parameters()),
            self.config.training.max_grad_norm
        )
        self.world_model_optimizer.step()
        
        # Update priorities
        with torch.no_grad():
            curr_latents = self.encoder(obs)
            next_pred = self.world_model(curr_latents, actions)['next_state']
            new_priorities = F.mse_loss(
                next_pred, target_next_latents, reduction='none'
            ).mean(dim=-1).cpu().numpy()
        
        self.replay_buffer.update_priorities(indices, new_priorities)
        self._soft_update_target_encoder()
        self.world_model_scheduler.step()
        
        return {
            'world_model/state_loss': losses['state_loss'].item(),
            'world_model/reward_loss': losses.get('reward_loss', torch.tensor(0.0)).item(),
            'world_model/done_loss': losses.get('done_loss', torch.tensor(0.0)).item(),
            'world_model/total_loss': weighted_loss.item()
        }
    
    def train_curiosity(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Train the curiosity module."""
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        if len(self.replay_buffer) < batch_size:
            return {}
        
        self.curiosity.train()
        self.rnd.predictor_net.train()
        
        experiences, weights, indices = self.replay_buffer.sample(batch_size)
        
        obs = torch.FloatTensor(np.array([e.obs for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        next_obs = torch.FloatTensor(np.array([e.next_obs for e in experiences])).to(self.device)
        
        with torch.no_grad():
            latents = self.encoder(obs)
            next_latents = self.encoder(next_obs)
        
        icm_output = self.curiosity(latents, actions, next_latents)
        icm_loss = icm_output['total_loss']
        
        rnd_output = self.rnd(next_latents)
        rnd_loss = rnd_output['loss']
        
        total_loss = icm_loss + 0.5 * rnd_loss
        
        self.curiosity_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.curiosity.parameters()) + list(self.rnd.predictor_net.parameters()),
            self.config.training.max_grad_norm
        )
        self.curiosity_optimizer.step()
        
        return {
            'curiosity/icm_forward_loss': icm_output['forward_loss'].item(),
            'curiosity/icm_inverse_loss': icm_output['inverse_loss'].item(),
            'curiosity/rnd_loss': rnd_loss.item(),
            'curiosity/total_loss': total_loss.item()
        }
    
    def train_policy_ppo(self) -> Dict[str, float]:
        """Train policy using PPO."""
        if self.rollout_buffer.ptr < self.config.training.batch_size:
            return {}
        
        self.policy.train()
        
        # Finish current path
        with torch.no_grad():
            last_obs = self.rollout_buffer.observations[self.rollout_buffer.ptr - 1]
            last_latent = self.encode(torch.FloatTensor(last_obs).to(self.device))
            last_value = self.policy(last_latent)['value'].item()
        
        self.rollout_buffer.finish_path(last_value)
        
        data = self.rollout_buffer.get()
        
        observations = data['observations'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        advantages = data['advantages'].to(self.device)
        returns = data['returns'].to(self.device)
        
        with torch.no_grad():
            latents = self.encoder(observations)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        train_config = self.config.training
        policy_config = self.config.policy
        
        batch_size = len(observations)
        mini_batch_size = train_config.mini_batch_size
        
        for _ in range(train_config.ppo_epochs):
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                mb_latents = latents[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                eval_output = self.policy.evaluate_actions(mb_latents, mb_actions)
                new_log_probs = eval_output['log_prob']
                values = eval_output['value']
                entropy = eval_output['entropy']
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - policy_config.clip_epsilon, 1 + policy_config.clip_epsilon
                ) * mb_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                
                loss = (
                    policy_loss +
                    policy_config.value_coef * value_loss +
                    policy_config.entropy_coef * entropy_loss
                )
                
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), policy_config.max_grad_norm
                )
                self.policy_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        self.policy_scheduler.step()
        self.rollout_buffer.reset()
        
        return {
            'policy/policy_loss': total_policy_loss / max(n_updates, 1),
            'policy/value_loss': total_value_loss / max(n_updates, 1),
            'policy/entropy': total_entropy / max(n_updates, 1)
        }
    
    def _soft_update_target_encoder(self, tau: float = 0.005):
        for target_param, param in zip(
            self.target_encoder.parameters(), self.encoder.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train_step(self) -> Dict[str, float]:
        """Single training step - updates all components."""
        metrics = {}
        
        for _ in range(self.config.training.world_model_updates_per_step):
            wm_metrics = self.train_world_model()
            metrics.update(wm_metrics)
        
        curiosity_metrics = self.train_curiosity()
        metrics.update(curiosity_metrics)
        
        policy_metrics = self.train_policy_ppo()
        metrics.update(policy_metrics)
        
        self.global_step += 1
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save complete agent state."""
        checkpoint = {
            'config': self.config,
            'global_step': self.global_step,
            'episodes_completed': self.episodes_completed,
            'best_reward': self.best_reward,
            'encoder_state_dict': self.encoder.state_dict(),
            'world_model_state_dict': self.world_model.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'curiosity_state_dict': self.curiosity.state_dict(),
            'rnd_state_dict': self.rnd.state_dict(),
            'target_encoder_state_dict': self.target_encoder.state_dict(),
            'world_model_optimizer_state_dict': self.world_model_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'curiosity_optimizer_state_dict': self.curiosity_optimizer.state_dict(),
            'metrics': dict(self.metrics)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.episodes_completed = checkpoint['episodes_completed']
        self.best_reward = checkpoint['best_reward']
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.curiosity.load_state_dict(checkpoint['curiosity_state_dict'])
        self.rnd.load_state_dict(checkpoint['rnd_state_dict'])
        self.target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
        
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.curiosity_optimizer.load_state_dict(checkpoint['curiosity_optimizer_state_dict'])
        
        self.metrics = defaultdict(list, checkpoint['metrics'])
        
        logger.info(f"Loaded checkpoint from {path}")


# PART 13: TRAINER

class MiniGridTrainer:
    """Complete training loop for MiniGrid environments."""
    def __init__(self, agent: UnifiedMiniGridAgent, config: AgentConfig,
                 log_dir: str = "./logs", checkpoint_dir: str = "./checkpoints"):
        self.agent = agent
        self.config = config
        
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create environments
        self.train_envs = VectorizedMiniGridEnv(
            env_name=config.env_name,
            n_envs=config.training.n_envs,
            use_full_obs=config.use_full_obs,
            use_rgb=config.use_rgb,
            seed=config.seed
        )
        
        self.eval_env = MiniGridWrapper(
            env_name=config.env_name,
            use_full_obs=config.use_full_obs,
            use_rgb=config.use_rgb
        )
        
        # Training state
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        
        # TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.warning("TensorBoard not available.")
    
    def train(self, total_timesteps: int = 1000000, eval_interval: int = 10000,
              checkpoint_interval: int = 50000, log_interval: int = 1000):
        """Main training loop."""
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        
        obs = self.train_envs.reset()
        n_envs = self.config.training.n_envs
        rollout_length = self.config.training.rollout_length
        
        episode_rewards = np.zeros(n_envs)
        episode_lengths = np.zeros(n_envs)
        
        while self.total_steps < total_timesteps:
            # Collect rollout
            for step in range(rollout_length):
                # Get actions
                action_output = self.agent.act(obs)
                actions = action_output['action']
                log_probs = action_output['log_prob']
                values = action_output['value']
                
                # Step environments
                next_obs, rewards, dones, infos = self.train_envs.step(actions)
                
                # Compute intrinsic rewards
                intrinsic_rewards = np.zeros(n_envs)
                for i in range(n_envs):
                    intrinsic_rewards[i] = self.agent.compute_intrinsic_reward(
                        obs[i], actions[i], next_obs[i]
                    )
                
                # Store experiences
                for i in range(n_envs):
                    self.agent.store_experience(
                        obs=obs[i],
                        action=actions[i],
                        reward=rewards[i],
                        next_obs=next_obs[i],
                        done=dones[i],
                        log_prob=log_probs[i],
                        value=values[i],
                        intrinsic_reward=intrinsic_rewards[i]
                    )
                
                # Track episodes
                episode_rewards += rewards
                episode_lengths += 1
                
                for i, done in enumerate(dones):
                    if done:
                        self.episode_rewards.append(episode_rewards[i])
                        self.episode_lengths.append(episode_lengths[i])
                        self.agent.episodes_completed += 1
                        episode_rewards[i] = 0
                        episode_lengths[i] = 0
                
                obs = next_obs
                self.total_steps += n_envs
            
            # Training updates
            train_metrics = self.agent.train_step()
            
            # Logging
            if self.total_steps % log_interval < n_envs * rollout_length:
                self._log_progress(train_metrics)
            
            # Evaluation
            if self.total_steps % eval_interval < n_envs * rollout_length:
                eval_reward = self._evaluate()
                self.eval_rewards.append(eval_reward)
                
                if eval_reward > self.agent.best_reward:
                    self.agent.best_reward = eval_reward
                    self._save_best_model()
            
            # Checkpointing
            if self.total_steps % checkpoint_interval < n_envs * rollout_length:
                self._save_checkpoint()
        
        logger.info("Training complete!")
        self._save_final_model()
        self.train_envs.close()
        self.eval_env.close()
    
    @torch.no_grad()
    def _evaluate(self, n_episodes: int = 10) -> float:
        """Evaluate agent without exploration."""
        total_reward = 0.0
        
        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0.0
            
            for _ in range(1000):
                action_output = self.agent.act(obs, deterministic=True)
                action = action_output['action']
                
                obs, reward, done, _, _ = self.eval_env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / n_episodes
    
    def _log_progress(self, train_metrics: Dict[str, float]):
        """Log training progress."""
        if len(self.episode_rewards) == 0:
            return
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)
        
        logger.info(
            f"Steps: {self.total_steps:8d} | "
            f"Episodes: {self.agent.episodes_completed:6d} | "
            f"Avg Reward: {avg_reward:8.2f} | "
            f"Best: {self.agent.best_reward:8.2f}"
        )
        
        if self.writer:
            self.writer.add_scalar('train/avg_reward', avg_reward, self.total_steps)
            self.writer.add_scalar('train/episodes', self.agent.episodes_completed, self.total_steps)
            
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, self.total_steps)
    
    def _save_checkpoint(self):
        path = self.checkpoint_dir / f"checkpoint_{self.total_steps}.pt"
        self.agent.save_checkpoint(str(path))
    
    def _save_best_model(self):
        path = self.checkpoint_dir / "best_model.pt"
        self.agent.save_checkpoint(str(path))
        logger.info(f"New best model! Reward: {self.agent.best_reward:.2f}")
    
    def _save_final_model(self):
        path = self.checkpoint_dir / "final_model.pt"
        self.agent.save_checkpoint(str(path))


# PART 14: VISUALIZATION AND ANALYSIS 

class MiniGridAnalyzer:
    """Tools for analyzing and visualizing agent behavior in MiniGrid."""
    def __init__(self, agent: UnifiedMiniGridAgent, env: MiniGridWrapper):
        self.agent = agent
        self.env = env
    
    def visualize_latent_space(self, n_episodes: int = 10, save_path: Optional[str] = None):
        """Visualize the learned latent space."""
        from sklearn.decomposition import PCA
        
        latents = []
        rewards = []
        step_indices = []
        episode_indices = []
        
        for ep in range(n_episodes):
            obs = self.env.reset()
            
            for step in range(200):
                with torch.no_grad():
                    latent = self.agent.encode(obs).cpu().numpy()[0]
                
                latents.append(latent)
                step_indices.append(step)
                episode_indices.append(ep)
                
                action_output = self.agent.act(obs, deterministic=False)
                action = action_output['action']
                
                next_obs, reward, done, _, _ = self.env.step(action)
                rewards.append(reward)
                
                if done:
                    break
                
                obs = next_obs
        
        latents = np.array(latents)
        rewards = np.array(rewards)
        step_indices = np.array(step_indices)
        episode_indices = np.array(episode_indices)
        
        # PCA reduction
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Color by reward
        scatter1 = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1],
                                   c=rewards, cmap='RdYlGn', s=20, alpha=0.6)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('Latent Space (Colored by Reward)')
        plt.colorbar(scatter1, ax=axes[0], label='Reward')
        
        # Plot 2: Color by step in episode
        scatter2 = axes[1].scatter(latents_2d[:, 0], latents_2d[:, 1],
                                   c=step_indices, cmap='viridis', s=20, alpha=0.6)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('Latent Space (Colored by Step)')
        plt.colorbar(scatter2, ax=axes[1], label='Step')
        
        # Plot 3: Color by episode
        scatter3 = axes[2].scatter(latents_2d[:, 0], latents_2d[:, 1],
                                   c=episode_indices, cmap='tab10', s=20, alpha=0.6)
        axes[2].set_xlabel('PC1')
        axes[2].set_ylabel('PC2')
        axes[2].set_title('Latent Space (Colored by Episode)')
        plt.colorbar(scatter3, ax=axes[2], label='Episode')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved latent space visualization to {save_path}")
        
        plt.show()
        
        return latents_2d, rewards, pca
    
    def visualize_world_model_predictions(self, n_steps: int = 50, 
                                           save_path: Optional[str] = None):
        """Visualize world model prediction accuracy."""
        from sklearn.decomposition import PCA
        
        actual_latents = []
        predicted_latents = []
        actions_taken = []
        prediction_errors = []
        
        obs = self.env.reset()
        
        with torch.no_grad():
            prev_latent = self.agent.encode(obs)
        
        for step in range(n_steps):
            action_output = self.agent.act(obs, deterministic=False)
            action = action_output['action']
            actions_taken.append(action)
            
            # Predict next state
            with torch.no_grad():
                prediction = self.agent.predict_next_state(obs, action)
                predicted_next_latent = prediction['next_latent'][0]
            
            # Take actual step
            next_obs, reward, done, _, _ = self.env.step(action)
            
            # Get actual next latent
            with torch.no_grad():
                actual_next_latent = self.agent.encode(next_obs).cpu().numpy()[0]
            
            actual_latents.append(actual_next_latent)
            predicted_latents.append(predicted_next_latent)
            
            # Compute prediction error
            error = np.linalg.norm(actual_next_latent - predicted_next_latent)
            prediction_errors.append(error)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        
        actual_latents = np.array(actual_latents)
        predicted_latents = np.array(predicted_latents)
        prediction_errors = np.array(prediction_errors)
        
        # PCA for visualization
        all_latents = np.vstack([actual_latents, predicted_latents])
        pca = PCA(n_components=2)
        all_latents_2d = pca.fit_transform(all_latents)
        
        actual_2d = all_latents_2d[:n_steps]
        predicted_2d = all_latents_2d[n_steps:]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Actual vs Predicted trajectories
        axes[0, 0].plot(actual_2d[:, 0], actual_2d[:, 1], 'b-o', 
                        label='Actual', alpha=0.7, markersize=4)
        axes[0, 0].plot(predicted_2d[:, 0], predicted_2d[:, 1], 'r--x', 
                        label='Predicted', alpha=0.7, markersize=4)
        
        # Draw lines connecting actual to predicted
        for i in range(n_steps):
            axes[0, 0].plot([actual_2d[i, 0], predicted_2d[i, 0]],
                           [actual_2d[i, 1], predicted_2d[i, 1]],
                           'gray', alpha=0.3, linewidth=0.5)
        
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        axes[0, 0].set_title('Imagination vs Reality: World Model Accuracy')
        axes[0, 0].legend()
        
        # Plot 2: Prediction error over time
        axes[0, 1].plot(prediction_errors, 'b-', linewidth=1.5)
        axes[0, 1].fill_between(range(len(prediction_errors)), prediction_errors, 
                                alpha=0.3)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Prediction Error (L2)')
        axes[0, 1].set_title('World Model Prediction Error Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        axes[1, 0].hist(prediction_errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(prediction_errors), color='r', linestyle='--',
                          label=f'Mean: {np.mean(prediction_errors):.4f}')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Distribution of Prediction Errors')
        axes[1, 0].legend()
        
        # Plot 4: Error by action
        action_errors = defaultdict(list)
        for a, e in zip(actions_taken, prediction_errors):
            action_errors[a].append(e)
        
        action_names = ['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
        x_positions = list(action_errors.keys())
        means = [np.mean(action_errors[a]) for a in x_positions]
        stds = [np.std(action_errors[a]) for a in x_positions]
        
        axes[1, 1].bar(x_positions, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1, 1].set_xlabel('Action')
        axes[1, 1].set_ylabel('Mean Prediction Error')
        axes[1, 1].set_title('Prediction Error by Action Type')
        axes[1, 1].set_xticks(x_positions)
        axes[1, 1].set_xticklabels([action_names[a] if a < len(action_names) else str(a) 
                                    for a in x_positions], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved world model predictions to {save_path}")
        
        plt.show()
        
        return {
            'mean_error': np.mean(prediction_errors),
            'std_error': np.std(prediction_errors),
            'max_error': np.max(prediction_errors),
            'min_error': np.min(prediction_errors)
        }
    
    def visualize_attention_patterns(self, obs: Optional[np.ndarray] = None,
                                      action: Optional[int] = None,
                                      save_path: Optional[str] = None):
        """Visualize attention patterns in the world model."""
        if obs is None:
            obs = self.env.reset()
        if action is None:
            action = np.random.randint(self.agent.config.action_dim)
        
        with torch.no_grad():
            latent = self.agent.encode(obs)
            action_t = torch.LongTensor([action]).to(self.agent.device)
            
            output = self.agent.world_model(latent, action_t, return_attention=True)
            attention = output['attention'].cpu().numpy()[0]  # [n_layers, n_heads, 2, 2]
        
        n_layers = attention.shape[0]
        n_heads = attention.shape[1]
        
        # Average across heads for each layer
        fig, axes = plt.subplots(1, n_layers + 1, figsize=(4 * (n_layers + 1), 4))
        
        labels = ['State', 'Action']
        
        for layer in range(n_layers):
            avg_attn = attention[layer].mean(axis=0)  # Average across heads
            
            im = axes[layer].imshow(avg_attn, cmap='Blues', vmin=0, vmax=1)
            axes[layer].set_xticks([0, 1])
            axes[layer].set_yticks([0, 1])
            axes[layer].set_xticklabels(labels)
            axes[layer].set_yticklabels(labels)
            axes[layer].set_title(f'Layer {layer + 1}')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    axes[layer].text(j, i, f'{avg_attn[i, j]:.3f}',
                                    ha='center', va='center', fontsize=12,
                                    color='white' if avg_attn[i, j] > 0.5 else 'black')
        
        # Average across all layers
        avg_all = attention.mean(axis=(0, 1))
        im = axes[n_layers].imshow(avg_all, cmap='Blues', vmin=0, vmax=1)
        axes[n_layers].set_xticks([0, 1])
        axes[n_layers].set_yticks([0, 1])
        axes[n_layers].set_xticklabels(labels)
        axes[n_layers].set_yticklabels(labels)
        axes[n_layers].set_title('Average All Layers')
        
        for i in range(2):
            for j in range(2):
                axes[n_layers].text(j, i, f'{avg_all[i, j]:.3f}',
                                   ha='center', va='center', fontsize=12,
                                   color='white' if avg_all[i, j] > 0.5 else 'black')
        
        plt.suptitle('Causal Linkage: Attention Map\n(Proof the World Model connects Action to Future)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved attention patterns to {save_path}")
        
        plt.show()
        
        # Print analysis
        state_to_action = avg_all[0, 1]
        action_to_state = avg_all[1, 0]
        
        logger.info(f"Attention Analysis:")
        logger.info(f"  State attending to Action: {state_to_action:.3f}")
        logger.info(f"  Action attending to State: {action_to_state:.3f}")
        
        if action_to_state > 0.5:
            logger.info("  -> Strong causal linkage: Model understands actions determine future states")
        
        return attention
    
    def visualize_curiosity_landscape(self, n_steps: int = 500,
                                       save_path: Optional[str] = None):
        """Visualize curiosity rewards across the state space."""
        from sklearn.decomposition import PCA
        
        latents = []
        curiosities = []
        positions = []
        
        obs = self.env.reset()
        
        for _ in range(n_steps):
            with torch.no_grad():
                latent = self.agent.encode(obs).cpu().numpy()[0]
            
            action_output = self.agent.act(obs, deterministic=False)
            action = action_output['action']
            
            next_obs, reward, done, _, _ = self.env.step(action)
            
            curiosity = self.agent.compute_intrinsic_reward(obs, action, next_obs)
            
            latents.append(latent)
            curiosities.append(curiosity)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        
        latents = np.array(latents)
        curiosities = np.array(curiosities)
        
        # PCA reduction
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Curiosity landscape
        scatter = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1],
                                  c=curiosities, cmap='hot', s=30, alpha=0.7)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title("The 'Fear' Map: Epistemic Uncertainty\n(Bright = Unknown, Dark = Known)")
        plt.colorbar(scatter, ax=axes[0], label='Curiosity (Model Confusion)')
        
        # Plot 2: Curiosity over time
        axes[1].plot(curiosities, 'b-', alpha=0.7, linewidth=0.8)
        axes[1].fill_between(range(len(curiosities)), curiosities, alpha=0.3)
        
        # Add rolling average
        window = 20
        if len(curiosities) > window:
            rolling_avg = np.convolve(curiosities, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(curiosities)), rolling_avg, 'r-', 
                        linewidth=2, label=f'Rolling Avg (window={window})')
            axes[1].legend()
        
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Curiosity Reward')
        axes[1].set_title('Curiosity Signal Over Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved curiosity landscape to {save_path}")
        
        plt.show()
        
        return latents_2d, curiosities
    
    def visualize_agent_trajectory(self, n_steps: int = 100,
                                    save_path: Optional[str] = None):
        """Visualize agent's trajectory with curiosity coloring."""
        from sklearn.decomposition import PCA
        
        observations = []
        latents = []
        actions = []
        rewards = []
        curiosities = []
        
        obs = self.env.reset()
        observations.append(obs.copy())
        
        for step in range(n_steps):
            with torch.no_grad():
                latent = self.agent.encode(obs).cpu().numpy()[0]
            latents.append(latent)
            
            action_output = self.agent.act(obs, deterministic=False)
            action = action_output['action']
            actions.append(action)
            
            next_obs, reward, done, _, _ = self.env.step(action)
            rewards.append(reward)
            
            curiosity = self.agent.compute_intrinsic_reward(obs, action, next_obs)
            curiosities.append(curiosity)
            
            observations.append(next_obs.copy())
            
            if done:
                break
            
            obs = next_obs
        
        latents = np.array(latents)
        curiosities = np.array(curiosities)
        
        # PCA reduction
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Trajectory with curiosity coloring
        scatter = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1],
                                  c=curiosities, cmap='plasma', s=50, alpha=0.8,
                                  zorder=2)
        
        # Draw trajectory lines
        for i in range(len(latents_2d) - 1):
            axes[0].plot([latents_2d[i, 0], latents_2d[i+1, 0]],
                        [latents_2d[i, 1], latents_2d[i+1, 1]],
                        'gray', alpha=0.5, linewidth=1, zorder=1)
        
        # Mark start and end
        axes[0].scatter(latents_2d[0, 0], latents_2d[0, 1], 
                       c='green', s=200, marker='*', label='Start', zorder=3)
        axes[0].scatter(latents_2d[-1, 0], latents_2d[-1, 1], 
                       c='red', s=200, marker='X', label='End', zorder=3)
        
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title(f'Agent Trajectory: {len(latents)} Steps\n(Hot = High Learning)')
        axes[0].legend()
        plt.colorbar(scatter, ax=axes[0], label='Curiosity (Surprise)')
        
        # Plot 2: Sample observations
        n_samples = min(6, len(observations))
        sample_indices = np.linspace(0, len(observations) - 1, n_samples, dtype=int)
        
        for idx, sample_idx in enumerate(sample_indices):
            ax_inset = axes[1].inset_axes([idx / n_samples, 0.1, 0.9 / n_samples, 0.8])
            obs_img = observations[sample_idx]
            
            # Handle different observation formats
            if obs_img.shape[-1] == 3:  # RGB
                ax_inset.imshow(obs_img.astype(np.uint8) if obs_img.max() > 1 else obs_img)
            else:
                ax_inset.imshow(obs_img[:, :, 0], cmap='gray')
            
            ax_inset.set_title(f'Step {sample_idx}', fontsize=8)
            ax_inset.axis('off')
        
        axes[1].set_title('Sample Observations Along Trajectory')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved agent trajectory to {save_path}")
        
        plt.show()
        
        return {
            'trajectory': latents_2d,
            'curiosities': curiosities,
            'rewards': rewards,
            'actions': actions
        }
    
    def visualize_imagination_rollout(self, horizon: int = 20,
                                       save_path: Optional[str] = None):
        """Visualize an imagined trajectory vs actual execution."""
        from sklearn.decomposition import PCA
        
        obs = self.env.reset()
        
        # Generate random action sequence
        action_sequence = [np.random.randint(self.agent.config.action_dim) 
                          for _ in range(horizon)]
        
        # Imagine trajectory
        imagined = self.agent.imagine_trajectory(obs, action_sequence)
        imagined_trajectory = imagined['trajectory'][0]  # [horizon+1, latent_dim]
        imagined_rewards = imagined['rewards'][0]
        imagined_uncertainties = imagined['uncertainties'][0]
        
        # Execute actual trajectory
        actual_latents = []
        actual_rewards = []
        
        with torch.no_grad():
            initial_latent = self.agent.encode(obs).cpu().numpy()[0]
        actual_latents.append(initial_latent)
        
        current_obs = obs.copy()
        for action in action_sequence:
            next_obs, reward, done, _, _ = self.env.step(action)
            actual_rewards.append(reward)
            
            with torch.no_grad():
                latent = self.agent.encode(next_obs).cpu().numpy()[0]
            actual_latents.append(latent)
            
            if done:
                # Pad with last state if episode ended
                while len(actual_latents) < horizon + 1:
                    actual_latents.append(latent)
                    actual_rewards.append(0)
                break
            
            current_obs = next_obs
        
        actual_latents = np.array(actual_latents)
        actual_rewards = np.array(actual_rewards)
        
        # Ensure same length
        min_len = min(len(actual_latents), len(imagined_trajectory))
        actual_latents = actual_latents[:min_len]
        imagined_trajectory = imagined_trajectory[:min_len]
        
        # PCA reduction
        all_latents = np.vstack([actual_latents, imagined_trajectory])
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_latents)
        
        actual_2d = all_2d[:min_len]
        imagined_2d = all_2d[min_len:]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Trajectories
        axes[0, 0].plot(actual_2d[:, 0], actual_2d[:, 1], 'b-o', 
                        label='Reality', alpha=0.8, markersize=6)
        axes[0, 0].plot(imagined_2d[:, 0], imagined_2d[:, 1], 'r--s', 
                        label='Imagination', alpha=0.8, markersize=6)
        
        # Mark start
        axes[0, 0].scatter(actual_2d[0, 0], actual_2d[0, 1], 
                          c='green', s=200, marker='*', zorder=5, label='Start')
        
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        axes[0, 0].set_title('Imagination vs Reality: World Model Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction error over horizon
        errors = np.linalg.norm(actual_latents - imagined_trajectory, axis=1)
        axes[0, 1].plot(errors, 'b-o', linewidth=2)
        axes[0, 1].fill_between(range(len(errors)), errors, alpha=0.3)
        axes[0, 1].set_xlabel('Step in Trajectory')
        axes[0, 1].set_ylabel('Prediction Error (L2)')
        axes[0, 1].set_title('World Model Error Accumulation Over Horizon')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Imagined uncertainties
        axes[1, 0].bar(range(len(imagined_uncertainties)), imagined_uncertainties,
                       color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Step in Trajectory')
        axes[1, 0].set_ylabel('Predicted Uncertainty')
        axes[1, 0].set_title('World Model Self-Assessed Uncertainty')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Correlation between uncertainty and actual error
        # Ensure same length
        min_unc_len = min(len(imagined_uncertainties), len(errors) - 1)
        unc_subset = imagined_uncertainties[:min_unc_len]
        err_subset = errors[1:min_unc_len + 1]  # Errors at next step
        
        axes[1, 1].scatter(unc_subset, err_subset, alpha=0.7, s=50)
        
        # Fit line
        if len(unc_subset) > 2:
            z = np.polyfit(unc_subset, err_subset, 1)
            p = np.poly1d(z)
            x_line = np.linspace(unc_subset.min(), unc_subset.max(), 100)
            axes[1, 1].plot(x_line, p(x_line), 'r--', label=f'Trend')
            
            # Correlation
            corr = np.corrcoef(unc_subset, err_subset)[0, 1]
            axes[1, 1].set_title(f'Uncertainty vs Actual Error (Corr: {corr:.3f})')
        else:
            axes[1, 1].set_title('Uncertainty vs Actual Error')
        
        axes[1, 1].set_xlabel('Predicted Uncertainty')
        axes[1, 1].set_ylabel('Actual Prediction Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved imagination rollout to {save_path}")
        
        plt.show()
        
        return {
            'actual_trajectory': actual_2d,
            'imagined_trajectory': imagined_2d,
            'errors': errors,
            'uncertainties': imagined_uncertainties
        }
    
    def analyze_policy_behavior(self, n_episodes: int = 20,
                                 save_path: Optional[str] = None):
        """Analyze policy behavior and action distribution."""
        action_counts = defaultdict(int)
        action_sequences = []
        episode_rewards = []
        episode_lengths = []
        state_values = []
        entropies = []
        
        action_names = ['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
        
        for ep in range(n_episodes):
            obs = self.env.reset()
            episode_actions = []
            episode_reward = 0
            
            for step in range(500):
                action_output = self.agent.act(obs, deterministic=False)
                action = action_output['action']
                
                action_counts[action] += 1
                episode_actions.append(action)
                state_values.append(action_output['value'])
                entropies.append(action_output['entropy'])
                
                next_obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
                
                obs = next_obs
            
            action_sequences.append(episode_actions)
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode_actions))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Action distribution
        actions = sorted(action_counts.keys())
        counts = [action_counts[a] for a in actions]
        labels = [action_names[a] if a < len(action_names) else str(a) for a in actions]
        
        axes[0, 0].bar(labels, counts, color='steelblue', alpha=0.8)
        axes[0, 0].set_xlabel('Action')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Action Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Episode rewards
        axes[0, 1].bar(range(len(episode_rewards)), episode_rewards, alpha=0.8)
        axes[0, 1].axhline(np.mean(episode_rewards), color='r', linestyle='--',
                          label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].set_title('Episode Rewards')
        axes[0, 1].legend()
        
        # Plot 3: Episode lengths
        axes[0, 2].bar(range(len(episode_lengths)), episode_lengths, 
                       color='green', alpha=0.8)
        axes[0, 2].axhline(np.mean(episode_lengths), color='r', linestyle='--',
                          label=f'Mean: {np.mean(episode_lengths):.1f}')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Steps')
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].legend()
        
        # Plot 4: Value estimates distribution
        axes[1, 0].hist(state_values, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(state_values), color='r', linestyle='--',
                          label=f'Mean: {np.mean(state_values):.3f}')
        axes[1, 0].set_xlabel('Value Estimate')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('State Value Distribution')
        axes[1, 0].legend()
        
        # Plot 5: Entropy over time
        axes[1, 1].plot(entropies, 'b-', alpha=0.5, linewidth=0.5)
        
        # Rolling average
        window = 100
        if len(entropies) > window:
            rolling = np.convolve(entropies, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(entropies)), rolling, 'r-', 
                           linewidth=2, label=f'Rolling Avg')
            axes[1, 1].legend()
        
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Policy Entropy')
        axes[1, 1].set_title('Policy Entropy Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Action transition matrix
        n_actions = max(action_counts.keys()) + 1
        transition_matrix = np.zeros((n_actions, n_actions))
        
        for seq in action_sequences:
            for i in range(len(seq) - 1):
                transition_matrix[seq[i], seq[i + 1]] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
        
        im = axes[1, 2].imshow(transition_matrix, cmap='Blues')
        axes[1, 2].set_xlabel('Next Action')
        axes[1, 2].set_ylabel('Current Action')
        axes[1, 2].set_title('Action Transition Probabilities')
        
        tick_labels = [action_names[i] if i < len(action_names) else str(i) 
                       for i in range(n_actions)]
        axes[1, 2].set_xticks(range(n_actions))
        axes[1, 2].set_yticks(range(n_actions))
        axes[1, 2].set_xticklabels(tick_labels, rotation=45)
        axes[1, 2].set_yticklabels(tick_labels)
        
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved policy analysis to {save_path}")
        
        plt.show()
        
        return {
            'action_counts': dict(action_counts),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_value': np.mean(state_values),
            'mean_entropy': np.mean(entropies)
        }
    
    def full_analysis(self, save_dir: str = "./analysis"):
        """Run complete analysis suite."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("RUNNING FULL AGENT ANALYSIS")
        logger.info("=" * 60)
        
        # 1. Latent space visualization
        logger.info("\n1. Analyzing latent space structure...")
        self.visualize_latent_space(n_episodes=10, 
                                    save_path=str(save_dir / "latent_space.png"))
        
        # 2. World model predictions
        logger.info("\n2. Analyzing world model predictions...")
        wm_stats = self.visualize_world_model_predictions(
            n_steps=100, save_path=str(save_dir / "world_model_predictions.png"))
        
        # 3. Attention patterns
        logger.info("\n3. Analyzing attention patterns...")
        self.visualize_attention_patterns(
            save_path=str(save_dir / "attention_patterns.png"))
        
        # 4. Curiosity landscape
        logger.info("\n4. Analyzing curiosity landscape...")
        self.visualize_curiosity_landscape(
            n_steps=300, save_path=str(save_dir / "curiosity_landscape.png"))
        
        # 5. Agent trajectory
        logger.info("\n5. Analyzing agent trajectory...")
        traj_stats = self.visualize_agent_trajectory(
            n_steps=100, save_path=str(save_dir / "agent_trajectory.png"))
        
        # 6. Imagination rollout
        logger.info("\n6. Analyzing imagination rollout...")
        imag_stats = self.visualize_imagination_rollout(
            horizon=15, save_path=str(save_dir / "imagination_rollout.png"))
        
        # 7. Policy behavior
        logger.info("\n7. Analyzing policy behavior...")
        policy_stats = self.analyze_policy_behavior(
            n_episodes=10, save_path=str(save_dir / "policy_analysis.png"))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)
        logger.info(f"\nWorld Model:")
        logger.info(f"  Mean Prediction Error: {wm_stats['mean_error']:.4f}")
        logger.info(f"  Std Prediction Error: {wm_stats['std_error']:.4f}")
        logger.info(f"\nPolicy:")
        logger.info(f"  Mean Episode Reward: {np.mean(policy_stats['episode_rewards']):.2f}")
        logger.info(f"  Mean Episode Length: {np.mean(policy_stats['episode_lengths']):.1f}")
        logger.info(f"  Mean State Value: {policy_stats['mean_value']:.3f}")
        logger.info(f"  Mean Entropy: {policy_stats['mean_entropy']:.3f}")
        logger.info(f"\nResults saved to: {save_dir}")
        
        return {
            'world_model': wm_stats,
            'trajectory': traj_stats,
            'imagination': imag_stats,
            'policy': policy_stats
        }


# PART 15: MODEL-BASED PLANNING

class ModelPredictiveControl:
    """
    Model Predictive Control (MPC) using the learned world model.
    
    Uses Cross-Entropy Method (CEM) to find optimal action sequences.
    """
    def __init__(self, agent: UnifiedMiniGridAgent, horizon: int = 10,
                 n_samples: int = 100, n_elite: int = 10, n_iterations: int = 5):
        self.agent = agent
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iterations = n_iterations
        self.action_dim = agent.config.action_dim
    
    @torch.no_grad()
    def plan(self, obs: np.ndarray) -> Tuple[int, Dict[str, np.ndarray]]:
        """
        Plan optimal action using CEM.
        
        Args:
            obs: Current observation
        
        Returns:
            best_action: First action of best sequence
            info: Dictionary with planning information
        """
        device = self.agent.device
        
        # Initialize uniform action distribution
        action_probs = torch.ones(self.horizon, self.action_dim, device=device) / self.action_dim
        
        all_rewards = []
        
        for iteration in range(self.n_iterations):
            # Sample action sequences
            action_sequences = torch.zeros(self.n_samples, self.horizon, 
                                           dtype=torch.long, device=device)
            
            for t in range(self.horizon):
                action_sequences[:, t] = torch.multinomial(
                    action_probs[t].unsqueeze(0).expand(self.n_samples, -1), 1
                ).squeeze(-1)
            
            # Evaluate sequences
            rewards = self._evaluate_sequences(obs, action_sequences)
            all_rewards.append(rewards.cpu().numpy())
            
            # Select elite sequences
            elite_indices = torch.argsort(rewards, descending=True)[:self.n_elite]
            elite_sequences = action_sequences[elite_indices]
            
            # Update action distribution
            for t in range(self.horizon):
                action_counts = torch.bincount(
                    elite_sequences[:, t], minlength=self.action_dim
                ).float()
                # Add smoothing
                action_probs[t] = (action_counts + 1) / (self.n_elite + self.action_dim)
        
        # Get best sequence from final iteration
        best_idx = elite_indices[0]
        best_sequence = action_sequences[best_idx].cpu().numpy()
        best_reward = rewards[best_idx].cpu().numpy()
        
        # Get planned trajectory for visualization
        latent = self.agent.encode(obs)
        trajectory = self.agent.world_model.imagine_trajectory(
            latent, action_sequences[best_idx].unsqueeze(0)
        )
        
        return best_sequence[0], {
            'best_sequence': best_sequence,
            'best_reward': best_reward,
            'trajectory': trajectory['trajectory'].cpu().numpy(),
            'uncertainties': trajectory['uncertainties'].cpu().numpy(),
            'all_rewards': all_rewards
        }
    
    def _evaluate_sequences(self, obs: np.ndarray, 
                            action_sequences: torch.Tensor) -> torch.Tensor:
        """Evaluate action sequences using world model."""
        n_samples = action_sequences.shape[0]
        
        # Encode initial state
        latent = self.agent.encode(obs)
        latent = latent.expand(n_samples, -1)
        
        # Imagine trajectories
        trajectories = self.agent.world_model.imagine_trajectory(latent, action_sequences)
        
        # Compute total reward with uncertainty penalty
        rewards = trajectories['rewards'].sum(dim=1)
        uncertainty_penalty = trajectories['uncertainties'].sum(dim=1) * 0.1
        
        # Bonus for reaching goals (high reward states)
        goal_bonus = (trajectories['rewards'].max(dim=1)[0] > 0.5).float() * 2.0
        
        return rewards - uncertainty_penalty + goal_bonus


class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search using the learned world model.
    """
    def __init__(self, agent: UnifiedMiniGridAgent, n_simulations: int = 100,
                 exploration_weight: float = 1.0, max_depth: int = 20):
        self.agent = agent
        self.n_simulations = n_simulations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.action_dim = agent.config.action_dim
    
    def search(self, obs: np.ndarray) -> Tuple[int, Dict]:
        """Perform MCTS and return best action."""
        root = MCTSNode(obs=obs, agent=self.agent)
        
        for _ in range(self.n_simulations):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not node.is_terminal:
                node = node.best_child(self.exploration_weight)
            
            # Expansion
            if not node.is_terminal and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation
            reward = self._simulate(node)
            
            # Backpropagation
            while node is not None:
                node.update(reward)
                node = node.parent
        
        # Get action with highest visit count
        best_action = root.best_action()
        
        # Collect statistics
        action_visits = {a: child.visits for a, child in root.children.items()}
        action_values = {a: child.value / max(child.visits, 1) 
                        for a, child in root.children.items()}
        
        return best_action, {
            'visits': action_visits,
            'values': action_values,
            'root_visits': root.visits
        }
    
    @torch.no_grad()
    def _simulate(self, node: 'MCTSNode') -> float:
        """Simulate from node using world model."""
        # Get latent state
        latent = self.agent.encode(node.obs)
        
        total_reward = 0.0
        discount = 1.0
        gamma = self.agent.config.training.gamma
        
        current_latent = latent
        
        for _ in range(self.max_depth):
            # Get action from policy
            action_output = self.agent.policy.get_action(current_latent, 
                                                          deterministic=False,
                                                          temperature=1.0)
            action = action_output['action']
            
            # Predict next state
            output = self.agent.world_model(current_latent, action, 
                                            return_uncertainty=True)
            
            # Accumulate reward
            pred_reward = output['predicted_reward'].item()
            total_reward += discount * pred_reward
            discount *= gamma
            
            # Check termination
            if output['predicted_done'].item() > 0.5:
                break
            
            current_latent = output['next_state']
        
        return total_reward


class MCTSNode:
    """Node in Monte Carlo Tree Search."""
    def __init__(self, obs: np.ndarray, agent: UnifiedMiniGridAgent,
                 parent: 'MCTSNode' = None, action: int = None):
        self.obs = obs
        self.agent = agent
        self.parent = parent
        self.action = action
        
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        
        self.untried_actions = list(range(agent.config.action_dim))
        random.shuffle(self.untried_actions)
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def expand(self) -> 'MCTSNode':
        """Expand by trying an untried action."""
        action = self.untried_actions.pop()
        
        # Predict next state using world model
        with torch.no_grad():
            prediction = self.agent.predict_next_state(self.obs, action)
        
        # For MCTS, we need to keep track of the latent state
        # Since we don't have the actual next observation, we'll store the latent
        child = MCTSNode(
            obs=self.obs,  # Note: in full implementation, decode or use env
            agent=self.agent,
            parent=self,
            action=action
        )
        
        child.is_terminal = prediction['predicted_done'][0] > 0.5
        child._latent = prediction['next_latent'][0]  # Store for simulation
        
        self.children[action] = child
        return child
    
    def best_child(self, exploration_weight: float) -> 'MCTSNode':
        """Select best child using UCB1."""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            exploit = child.value / max(child.visits, 1)
            explore = exploration_weight * np.sqrt(
                np.log(max(self.visits, 1)) / max(child.visits, 1)
            )
            score = exploit + explore
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def best_action(self) -> int:
        """Return action with highest visit count."""
        return max(self.children.keys(), 
                   key=lambda a: self.children[a].visits)
    
    def update(self, reward: float):
        """Update node statistics."""
        self.visits += 1
        self.value += reward


# PART 16: MAIN ENTRY POINT

def create_minigrid_config(env_name: str = "MiniGrid-Empty-8x8-v0",
                           use_rgb: bool = True) -> AgentConfig:
    """Create configuration for MiniGrid environment."""
    
    # Create temporary environment to get observation shape
    temp_env = MiniGridWrapper(env_name, use_full_obs=False, use_rgb=use_rgb)
    obs_shape = temp_env.get_obs_shape()
    action_dim = temp_env.get_action_dim()
    temp_env.close()
    
    return AgentConfig(
        obs_shape=obs_shape,
        latent_dim=256,
        action_dim=action_dim,
        cnn=CNNEncoderConfig(
            input_channels=obs_shape[-1],
            hidden_channels=[32, 64, 128],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
            use_batch_norm=True,
            use_residual=True,
            dropout=0.1
        ),
        transformer=TransformerConfig(
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            dropout=0.1,
            max_seq_len=256
        ),
        policy=PolicyConfig(
            hidden_dim=256,
            n_layers=3,
            n_heads=4,
            dropout=0.1,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5
        ),
        curiosity=CuriosityConfig(
            hidden_dim=256,
            feature_dim=128,
            forward_loss_coef=0.2,
            inverse_loss_coef=0.8,
            intrinsic_reward_scale=0.1
        ),
        training=TrainingConfig(
            learning_rate=2.5e-4,
            batch_size=256,
            mini_batch_size=64,
            ppo_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            n_envs=8,
            rollout_length=128,
            world_model_updates_per_step=4
        ),
        buffer=BufferConfig(
            capacity=100000,
            prioritized=True
        ),
        env_name=env_name,
        use_full_obs=False,
        use_rgb=use_rgb,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42
    )


def main():
    """Main entry point for MiniGrid unified agent."""
    
    if not MINIGRID_AVAILABLE:
        print("=" * 60)
        print("ERROR: MiniGrid not installed!")
        print("Please install with: pip install minigrid")
        print("=" * 60)
        return
    
    print("=" * 80)
    print("UNIFIED AGENT FOR MINIGRID")
    print("Transformer World Model + Curiosity-Driven RL")
    print("=" * 80)
    print()
    print("Architecture:")
    print("  - CNN State Encoder: Visual Perception")
    print("  - Transformer World Model: Reasoning about dynamics")
    print("  - Policy Network: Agency and decision making")
    print("  - Curiosity Module: Intrinsic motivation")
    print()
    print("The agent learns to UNDERSTAND through visual interaction.")
    print("=" * 80)
    print()
    
    # Select environment
    env_name = "MiniGrid-Empty-8x8-v0"
    
    # Available MiniGrid environments:
    # - MiniGrid-Empty-5x5-v0, MiniGrid-Empty-8x8-v0, MiniGrid-Empty-16x16-v0
    # - MiniGrid-FourRooms-v0
    # - MiniGrid-DoorKey-5x5-v0, MiniGrid-DoorKey-8x8-v0
    # - MiniGrid-MultiRoom-N2-S4-v0
    # - MiniGrid-KeyCorridorS3R1-v0
    # - MiniGrid-SimpleCrossingS9N1-v0
    
    print(f"Environment: {env_name}")
    print()
    
    # Create configuration
    print("Creating configuration...")
    config = create_minigrid_config(env_name=env_name, use_rgb=True)
    print(f"  Observation shape: {config.obs_shape}")
    print(f"  Action dimension: {config.action_dim}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Device: {config.device}")
    print()
    
    # Create agent
    print("Initializing agent...")
    agent = UnifiedMiniGridAgent(config)
    print()
    
    # Create trainer
    print("Setting up trainer...")
    trainer = MiniGridTrainer(
        agent=agent,
        config=config,
        log_dir="./logs_minigrid",
        checkpoint_dir="./checkpoints_minigrid"
    )
    print()
    
    # Train
    print("Starting training...")
    print("-" * 80)
    
    # For demonstration, use fewer timesteps
    # For full training, use 1_000_000+ timesteps
    total_timesteps = 100_000  # Adjust based on compute budget
    
    trainer.train(
        total_timesteps=total_timesteps,
        eval_interval=10000,
        checkpoint_interval=25000,
        log_interval=2000
    )
    
    print("-" * 80)
    print()
    
    # Analysis
    print("Running analysis...")
    print("=" * 80)
    
    eval_env = MiniGridWrapper(
        env_name=config.env_name,
        use_full_obs=config.use_full_obs,
        use_rgb=config.use_rgb
    )
    
    analyzer = MiniGridAnalyzer(agent, eval_env)
    analysis_results = analyzer.full_analysis(save_dir="./analysis_minigrid")
    
    eval_env.close()
    
    # Demonstrate understanding
    print()
    print("=" * 80)
    print("DEMONSTRATING UNDERSTANDING")
    print("=" * 80)
    
    demo_env = MiniGridWrapper(
        env_name=config.env_name,
        use_full_obs=config.use_full_obs,
        use_rgb=config.use_rgb
    )
    
    # Test 1: World model prediction accuracy
    print("\n1. World Model Prediction Accuracy")
    print("-" * 40)
    
    obs = demo_env.reset()
    errors = []
    
    for _ in range(50):
        action_output = agent.act(obs, deterministic=True)
        action = action_output['action']
        
        # Predict
        prediction = agent.predict_next_state(obs, action)
        predicted_latent = prediction['next_latent'][0]
        
        # Actual
        next_obs, reward, done, _, _ = demo_env.step(action)
        
        with torch.no_grad():
            actual_latent = agent.encode(next_obs).cpu().numpy()[0]
        
        error = np.linalg.norm(predicted_latent - actual_latent)
        errors.append(error)
        
        if done:
            obs = demo_env.reset()
        else:
            obs = next_obs
    
    print(f"  Mean prediction error: {np.mean(errors):.4f}")
    print(f"  Std prediction error: {np.std(errors):.4f}")
    
    # Test 2: Planning capability
    print("\n2. Planning Capability (MPC)")
    print("-" * 40)
    
    mpc = ModelPredictiveControl(agent, horizon=10, n_samples=50, 
                                  n_elite=5, n_iterations=3)
    
    obs = demo_env.reset()
    planned_rewards = []
    actual_rewards = []
    
    for step in range(20):
        # Plan
        best_action, plan_info = mpc.plan(obs)
        
        # Execute
        next_obs, reward, done, _, _ = demo_env.step(best_action)
        actual_rewards.append(reward)
        planned_rewards.append(plan_info['best_reward'])
        
        if done:
            break
        
        obs = next_obs
    
    print(f"  Planned total reward: {sum(planned_rewards):.2f}")
    print(f"  Actual total reward: {sum(actual_rewards):.2f}")
    print(f"  Planning horizon: 10 steps")
    
    # Test 3: Curiosity-driven exploration
    print("\n3. Curiosity-Driven Exploration")
    print("-" * 40)
    
    obs = demo_env.reset()
    curiosities = []
    
    for _ in range(30):
        action_output = agent.act(obs, deterministic=False, temperature=1.0)
        action = action_output['action']
        
        next_obs, reward, done, _, _ = demo_env.step(action)
        
        curiosity = agent.compute_intrinsic_reward(obs, action, next_obs)
        curiosities.append(curiosity)
        
        if done:
            obs = demo_env.reset()
        else:
            obs = next_obs
    
    print(f"  Mean curiosity reward: {np.mean(curiosities):.4f}")
    print(f"  Std curiosity: {np.std(curiosities):.4f}")
    print(f"  Max curiosity: {np.max(curiosities):.4f}")
    
    demo_env.close()
    
    
    print("=" * 80)
    print("Training and demonstration complete!")
    print("=" * 80)
    print()
    print("Files saved:")
    print("  - Checkpoints: ./checkpoints_minigrid/")
    print("  - Logs: ./logs_minigrid/")
    print("  - Analysis: ./analysis_minigrid/")


def quick_demo():
    """Quick demonstration without full training."""
    
    if not MINIGRID_AVAILABLE:
        print("MiniGrid not installed. Run: pip install minigrid")
        return
    
    print("=" * 60)
    print("QUICK DEMO: Unified Agent for MiniGrid")
    print("=" * 60)
    
    # Create config and agent
    config = create_minigrid_config("MiniGrid-Empty-5x5-v0")
    agent = UnifiedMiniGridAgent(config)
    
    # Create environment
    env = MiniGridWrapper(config.env_name, use_rgb=config.use_rgb)
    
    # Run a few episodes to collect some experience
    print("\nCollecting initial experience...")
    
    for ep in range(5):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(100):
            action_output = agent.act(obs, deterministic=False, temperature=2.0)
            action = action_output['action']
            
            next_obs, reward, done, _, _ = env.step(action)
            
            intrinsic = agent.compute_intrinsic_reward(obs, action, next_obs)
            
            agent.store_experience(
                obs=obs, action=action, reward=reward, next_obs=next_obs,
                done=done, log_prob=action_output['log_prob'],
                value=action_output['value'], intrinsic_reward=intrinsic
            )
            
            episode_reward += reward
            
            if done:
                break
            
            obs = next_obs
        
        # Train
        if len(agent.replay_buffer) >= config.training.batch_size:
            agent.train_step()
        
        print(f"  Episode {ep + 1}: reward = {episode_reward:.2f}, steps = {step + 1}")
    
    # Quick analysis
    print("\nRunning quick analysis...")
    
    analyzer = MiniGridAnalyzer(agent, env)
    
    # Just run attention visualization
    print("\n1. Attention Pattern Analysis:")
    obs = env.reset()
    action = 2  # Forward
    analyzer.visualize_attention_patterns(obs, action)
    
    # Quick trajectory
    print("\n2. Sample Trajectory:")
    obs = env.reset()
    print(f"   Starting exploration...")
    
    for step in range(10):
        action_output = agent.act(obs, deterministic=False)
        action = action_output['action']
        next_obs, reward, done, _, _ = env.step(action)
        
        action_names = ['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Done']
        print(f"   Step {step + 1}: {action_names[action]}, reward = {reward}")
        
        if done:
            print("   -> Reached goal!")
            break
        
        obs = next_obs
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Quick demo complete!")
    print("For full training, run main()")
    print("=" * 60)


if __name__ == "__main__":
        main()