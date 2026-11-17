import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)

        position = torch.arange(
            seq_len,
            device=x.device,
            dtype=x.dtype
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0, self.d_model, 2,
                device=x.device,
                dtype=x.dtype
            ) * (-math.log(10000.0) / self.d_model)
        )

        pe = torch.zeros(
            seq_len,
            self.d_model,
            device=x.device,
            dtype=x.dtype
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        use_cls=True
    ):
        super().__init__()

        assert d_model % nhead == 0

        self.use_cls = use_cls

        self.input_projection = (
            nn.Identity()
            if input_dim == d_model
            else nn.Linear(input_dim, d_model)
        )

        if use_cls:
            self.cls_token = nn.Parameter(
                torch.randn(1, 1, d_model)
            )

        self.positional_encoding = (
            SinusoidalPositionalEncoding(d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, lengths):
        """
        x:       (B, T, input_dim)
        lengths: (B,)

        returns: (B, d_model)
        """

        B, T, _ = x.shape

        # Embeddings -> d_model
        x = self.input_projection(x)

        # Padding mask for the original sequence
        padding_mask = (
            torch.arange(T, device=x.device).unsqueeze(0)
            >= lengths.unsqueeze(1)
        )

        if self.use_cls:

            # (1, 1, d_model) -> (B, 1, d_model)
            cls = self.cls_token.expand(B, -1, -1)

            # CLS is placed before the sequence
            x = torch.cat([cls, x], dim=1)

            # CLS is always valid
            cls_mask = torch.zeros(
                B, 1,
                dtype=torch.bool,
                device=x.device
            )

            padding_mask = torch.cat(
                [cls_mask, padding_mask],
                dim=1
            )

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        if self.use_cls:
            # CLS representation
            return x[:, 0]

        else:
            # Last valid time step
            batch_idx = torch.arange(B, device=x.device)
            return x[batch_idx, lengths - 1]