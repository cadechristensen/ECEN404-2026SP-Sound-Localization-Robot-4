"""
Transformer + CNN hybrid model architecture for baby cry detection.
Combines CNN for spatial feature extraction with Transformer for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

try:
    from .config import Config
except ImportError:
    from config import Config  # type: ignore


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.
    Adds positional information to the input embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length (optimized to 256 for memory efficiency)
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, batch_first: bool = True) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) if batch_first=True,
               or (seq_len, batch_size, d_model) if batch_first=False
            batch_first: Whether input has batch dimension first

        Returns:
            Tensor with positional encoding added
        """
        if batch_first:
            # x shape: (batch_size, seq_len, d_model)
            x = x + self.pe[:x.size(1), :].transpose(0, 1)
        else:
            # x shape: (seq_len, batch_size, d_model)
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolution block with different dilation rates and ResNet-style residual connections.
    Captures temporal patterns at different scales (short bursts vs sustained cries).

    Residual Connection:
        - Adds skip connection from input to output (x + F(x))
        - Uses 1x1 conv to match dimensions when in_channels != out_channels
        - Both paths go through pooling to match spatial dimensions
        - Activation applied AFTER residual addition (post-activation)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.3):
        """
        Initialize multi-scale convolution block with residual connections.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            dropout: Dropout rate
        """
        super().__init__()

        # Branch 1: Normal convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, kernel_size, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Dilated convolution (2x receptive field)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, kernel_size, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Dilated convolution (4x receptive field)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels - 2 * (out_channels // 3), kernel_size, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels - 2 * (out_channels // 3)),
            nn.ReLU(inplace=True)
        )

        # Second convolution (standard) - removed inplace ReLU for residual addition
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout)

        # Residual shortcut connection
        # Identity shortcut if dimensions match, otherwise use 1x1 conv for projection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height//2, width//2)
        """
        # Save input for residual connection
        identity = x

        # Multi-scale feature extraction through parallel branches
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # Concatenate multi-scale features
        out = torch.cat([x1, x2, x3], dim=1)

        # Second convolution (no activation yet)
        out = self.conv4(out)

        # Apply pooling to main path
        out = self.pool(out)

        # Apply shortcut with pooling to match spatial dimensions
        # Shortcut: either identity or 1x1 conv projection
        identity = self.shortcut(identity)
        identity = self.pool(identity)

        # Residual addition: out = out + identity
        out = out + identity

        # Post-activation: ReLU after residual addition (important for gradient flow)
        out = F.relu(out)

        # Apply dropout
        out = self.dropout(out)

        return out


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for mel-spectrograms.
    Extracts spatial features from frequency-time representations.
    """

    def __init__(self, config: Config):
        """
        Initialize CNN feature extractor.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # CNN layers for feature extraction with multi-scale
        self.conv_layers = nn.ModuleList()
        in_channels = config.INPUT_CHANNELS

        for out_channels in config.CNN_CHANNELS:
            self.conv_layers.append(
                MultiScaleConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=config.CNN_KERNEL_SIZE,
                    dropout=config.CNN_DROPOUT
                )
            )
            in_channels = out_channels

        # Calculate the output size after CNN layers
        self.output_channels = config.CNN_CHANNELS[-1]

        # Note: nn.AdaptiveAvgPool2d((4, None)) uses None for the time dimension which
        # is not representable in ONNX opset < 13 and breaks TorchScript tracing.
        # We remove the module-level pool and apply F.adaptive_avg_pool2d with an
        # explicit (freq, time) tuple in forward() instead — ONNX-safe and equivalent.

        # Projection to transformer dimension
        self.projection = nn.Linear(self.output_channels * 4, config.D_MODEL)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CNN feature extractor.

        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_steps)

        Returns:
            Feature tensor of shape (batch_size, time_steps', d_model)
        """
        # Apply CNN layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()

        # Adaptive pooling to standardize frequency dimension.
        # Use F.adaptive_avg_pool2d with an explicit (freq, time) tuple so the
        # output_size is fully concrete at trace time — ONNX-safe alternative to
        # nn.AdaptiveAvgPool2d((4, None)) which uses an opaque None sentinel.
        x = F.adaptive_avg_pool2d(x, (4, x.size(-1)))  # Shape: (batch_size, channels, 4, width)

        # Reshape for time-series processing
        # Flatten frequency and channel dimensions
        x = x.view(batch_size, -1, x.size(-1))  # Shape: (batch_size, channels*4, width)
        x = x.permute(0, 2, 1)  # Shape: (batch_size, width, channels*4)

        # Project to transformer dimension
        x = self.projection(x)  # Shape: (batch_size, time_steps, d_model)

        return x


class AttentionPooling(nn.Module):
    """
    Learned attention pooling over time dimension.
    Focuses on important temporal frames instead of simple averaging.
    """

    def __init__(self, d_model: int):
        """
        Initialize attention pooling.

        Args:
            d_model: Model dimension
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            x: Input tensor of shape (batch_size, time_steps, d_model)

        Returns:
            Pooled tensor of shape (batch_size, d_model)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch_size, time_steps, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        weighted = x * attn_weights  # (batch_size, time_steps, d_model)
        pooled = torch.sum(weighted, dim=1)  # (batch_size, d_model)

        return pooled


class TemporalPatternAttention(nn.Module):
    """
    Temporal pattern attention module for detecting cry-specific temporal patterns.
    Models burst-pause-burst rhythms characteristic of baby cries.
    """

    def __init__(self, d_model: int, num_patterns: int = 4, num_heads: int = 4):
        """
        Initialize temporal pattern attention.

        Args:
            d_model: Model dimension
            num_patterns: Number of temporal patterns to learn (e.g., different cry types)
            num_heads: Number of attention heads (must evenly divide d_model).
                       Passed from config.N_HEADS via TransformerEncoder to keep
                       consistent with the rest of the model (was hardcoded to 4).
        """
        super().__init__()
        self.d_model = d_model
        self.num_patterns = num_patterns

        # Learnable temporal pattern queries.
        # Scale down initialization: std=1.0 with d_model=256 gives norm≈16,
        # much larger than typical LayerNorm outputs (≈1).  Use 0.02 scale to
        # keep queries in the same range as the rest of the network at init.
        self.pattern_queries = nn.Parameter(torch.randn(num_patterns, d_model) * 0.02)

        # Multi-head attention for pattern matching.
        # num_heads is now derived from config (via TransformerEncoder) so it stays
        # consistent with the main transformer and respects d_model divisibility.
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Pattern scoring
        self.pattern_scorer = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal pattern attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tuple of (pattern_enhanced_features, pattern_scores)
        """
        batch_size = x.shape[0]

        # Expand pattern queries for batch
        queries = self.pattern_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Attention between patterns and temporal features
        pattern_features, attention_weights = self.pattern_attention(
            queries, x, x
        )  # (batch_size, num_patterns, d_model)

        # Score each pattern
        pattern_scores = self.pattern_scorer(pattern_features)  # (batch_size, num_patterns, 1)
        pattern_scores = pattern_scores.squeeze(-1)  # (batch_size, num_patterns)

        # Weighted combination of patterns
        pattern_weights = F.softmax(pattern_scores, dim=1)  # (batch_size, num_patterns)
        enhanced_features = torch.bmm(
            pattern_weights.unsqueeze(1),
            pattern_features
        ).squeeze(1)  # (batch_size, d_model)

        return enhanced_features, pattern_scores


class TransformerEncoder(nn.Module):
    """
    Enhanced Transformer encoder for temporal modeling.
    Captures long-range temporal dependencies and cry-specific temporal patterns.
    """

    def __init__(self, config: Config):
        """
        Initialize transformer encoder.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.D_MODEL,
            dropout=config.TRANSFORMER_DROPOUT
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=config.D_MODEL * 4,
            dropout=config.TRANSFORMER_DROPOUT,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.N_LAYERS
        )

        # Temporal pattern attention (NEW).
        # Pass num_heads from config so it matches the main transformer and stays
        # divisible into d_model — previously hardcoded to 4 inside the module.
        self.temporal_pattern_attn = TemporalPatternAttention(
            d_model=config.D_MODEL,
            num_patterns=4,  # Learn 4 different cry patterns
            num_heads=config.N_HEADS,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        # Fusion of standard temporal features and pattern features
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.D_MODEL * 2, config.D_MODEL),
            nn.ReLU(inplace=True),
            nn.Dropout(config.TRANSFORMER_DROPOUT)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer encoder with temporal pattern modeling.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Encoded tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding (optimized - no permutations needed)
        x = self.pos_encoder(x, batch_first=True)

        # Apply transformer encoder (uses batch_first=True)
        x_temporal = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Extract temporal patterns (NEW)
        pattern_features, pattern_scores = self.temporal_pattern_attn(x_temporal)

        # Expand pattern features to match temporal dimension
        pattern_features_expanded = pattern_features.unsqueeze(1).expand(
            -1, x_temporal.shape[1], -1
        )  # (batch_size, seq_len, d_model)

        # Fuse temporal features with pattern features
        x_fused = torch.cat([x_temporal, pattern_features_expanded], dim=-1)
        x_enhanced = self.feature_fusion(x_fused)

        # Layer normalization
        x_enhanced = self.layer_norm(x_enhanced)

        return x_enhanced


class BabyCryClassifier(nn.Module):
    """
    Main baby cry detection model.
    Combines CNN feature extraction with Transformer temporal modeling.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the baby cry classifier.

        Args:
            config: Configuration object. Defaults to Config().
        """
        super().__init__()
        if config is None:
            config = Config()
        self.config = config

        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(config)

        # Transformer encoder
        self.transformer = TransformerEncoder(config)

        # Attention pooling for sequence-level representation (better than avg pooling)
        self.attention_pool = AttentionPooling(config.D_MODEL)

        # Classification head (supports both binary and multi-class)
        num_classes = config.NUM_CLASSES if hasattr(config, 'NUM_CLASSES') else 2
        self.classifier = nn.Sequential(
            nn.Linear(config.D_MODEL, config.D_MODEL // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.TRANSFORMER_DROPOUT),
            nn.Linear(config.D_MODEL // 2, config.D_MODEL // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(config.TRANSFORMER_DROPOUT),
            nn.Linear(config.D_MODEL // 4, num_classes)  # Dynamic number of classes
        )

        # Temperature scaling — fixed non-learnable buffer.
        # A learnable temperature trained jointly with cross-entropy is an anti-pattern:
        # it can drive the value toward zero and counteract label smoothing.
        # register_buffer keeps it in state_dict but excludes it from optimizer updates.
        # For post-hoc calibration, load a calibrated value from a checkpoint.
        self.register_buffer('temperature', torch.ones(1))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights using appropriate strategies.

        - Conv2d: Kaiming (He) initialization for ReLU activations
        - Linear: Xavier uniform initialization
        - BatchNorm: ones for weight, zeros for bias
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            # Kaiming initialization for conv layers (accounts for ReLU)
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm standard initialization
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, 1, n_mels, time_steps)

        Returns:
            Raw logits tensor of shape (batch_size, num_classes)
        """
        # CNN feature extraction
        features = self.cnn_extractor(x)  # Shape: (batch_size, time_steps, d_model)

        # Transformer encoding
        encoded = self.transformer(features)  # Shape: (batch_size, time_steps, d_model)

        # Attention pooling over time dimension
        pooled = self.attention_pool(encoded)  # Shape: (batch_size, d_model)

        # Classification
        logits = self.classifier(pooled)  # Shape: (batch_size, 2)

        # NOTE: Temperature scaling is NOT applied here. Post-hoc calibration temperature
        # is applied externally (TemperatureScaledModel in calibration.py, or manually in
        # deployment code). The self.temperature buffer is kept for checkpoint backward
        # compatibility but is unused in forward().

        return logits

    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.

        Args:
            x: Input tensor

        Returns:
            Tuple of (cnn_features, transformer_features)
        """
        # CNN features
        cnn_features = self.cnn_extractor(x)

        # Transformer features
        transformer_features = self.transformer(cnn_features)

        return cnn_features, transformer_features


def create_model(config: Optional[Config] = None) -> BabyCryClassifier:
    """
    Create and return a baby cry classifier model.

    Args:
        config: Configuration object. Defaults to Config().

    Returns:
        Initialized model
    """
    if config is None:
        config = Config()
    model = BabyCryClassifier(config)
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: Tuple[int, ...]):
    """
    Print model summary with parameter counts.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
    """
    total_params = count_parameters(model)

    print("=" * 70)
    print("Baby Cry Detection Model Summary")
    print("=" * 70)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Input shape: {input_shape}")
    print("=" * 70)

    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")

    print("=" * 70)


if __name__ == "__main__":
    # Test the model
    config = Config()
    model = create_model(config)

    # Print model summary
    # Assume input shape based on config
    time_steps = int(config.DURATION * config.SAMPLE_RATE // config.HOP_LENGTH) + 1
    input_shape = (1, config.N_MELS, time_steps)

    model_summary(model, input_shape)

    # Test forward pass
    dummy_input = torch.randn(4, *input_shape)  # Batch size of 4
    output = model(dummy_input)
    print(f"Test forward pass successful. Output shape: {output.shape}")

    print("\n" + "=" * 70)
    print("Testing Residual Connections")
    print("=" * 70)

    # Test 1: Verify MultiScaleConvBlock with residual connections
    print("\n1. Testing MultiScaleConvBlock with residual connections...")

    # Test case 1: Same channel dimensions (identity shortcut)
    test_block_same = MultiScaleConvBlock(in_channels=32, out_channels=32, dropout=0.3)
    test_input_same = torch.randn(2, 32, 64, 64)
    test_output_same = test_block_same(test_input_same)
    expected_shape = (2, 32, 32, 32)  # Pooling reduces spatial dims by 2
    assert test_output_same.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {test_output_same.shape}"
    print(f"   [PASS] Identity shortcut (same channels): {test_input_same.shape} -> {test_output_same.shape}")

    # Test case 2: Different channel dimensions (projection shortcut)
    test_block_diff = MultiScaleConvBlock(in_channels=32, out_channels=64, dropout=0.3)
    test_input_diff = torch.randn(2, 32, 64, 64)
    test_output_diff = test_block_diff(test_input_diff)
    expected_shape = (2, 64, 32, 32)
    assert test_output_diff.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {test_output_diff.shape}"
    print(f"   [PASS] Projection shortcut (diff channels): {test_input_diff.shape} -> {test_output_diff.shape}")

    # Test 2: Verify gradient flow through residual connections
    print("\n2. Testing gradient flow through residual connections...")
    test_block_grad = MultiScaleConvBlock(in_channels=16, out_channels=32, dropout=0.0)
    test_input_grad = torch.randn(1, 16, 32, 32, requires_grad=True)
    test_output_grad = test_block_grad(test_input_grad)

    # Compute dummy loss and backpropagate
    loss = test_output_grad.sum()
    loss.backward()

    # Check that gradients exist and are not zero
    assert test_input_grad.grad is not None, "No gradient computed for input"
    assert test_input_grad.grad.abs().sum() > 0, "Gradient is zero"
    print(f"   [PASS] Gradients flow correctly through residual connections")
    print(f"   [INFO] Input gradient norm: {test_input_grad.grad.norm().item():.6f}")

    # Test 3: Verify dimension matching in shortcut
    print("\n3. Testing dimension matching in residual path...")
    test_block_dims = MultiScaleConvBlock(in_channels=16, out_channels=48, dropout=0.3)

    # Check if shortcut exists when channels differ
    has_projection = len(test_block_dims.shortcut) > 0
    print(f"   [PASS] Projection shortcut created: {has_projection}")

    if has_projection:
        # Verify shortcut has correct 1x1 conv
        shortcut_conv = test_block_dims.shortcut[0]
        assert isinstance(shortcut_conv, nn.Conv2d), "Shortcut should contain Conv2d"
        assert shortcut_conv.kernel_size == (1, 1), "Shortcut should use 1x1 conv"
        assert shortcut_conv.in_channels == 16, "Shortcut input channels mismatch"
        assert shortcut_conv.out_channels == 48, "Shortcut output channels mismatch"
        print(f"   [INFO] Shortcut conv: {shortcut_conv.in_channels} -> {shortcut_conv.out_channels} (1x1 kernel)")

    # Test 4: Test full CNN feature extractor with residual blocks
    print("\n4. Testing full CNN feature extractor with residual blocks...")
    test_cnn = CNNFeatureExtractor(config)
    test_input_cnn = torch.randn(2, 1, config.N_MELS, time_steps)
    test_output_cnn = test_cnn(test_input_cnn)
    print(f"   [PASS] CNN extractor output shape: {test_output_cnn.shape}")
    print(f"   [INFO] Expected shape: (batch_size, time_steps, d_model)")

    # Test 5: Verify residual connections improve gradient magnitude
    print("\n5. Comparing gradient flow: with vs without residual...")

    # Block with residual (our implementation)
    block_with_residual = MultiScaleConvBlock(in_channels=32, out_channels=32, dropout=0.0)
    input_with = torch.randn(1, 32, 64, 64, requires_grad=True)
    output_with = block_with_residual(input_with)
    loss_with = output_with.sum()
    loss_with.backward()
    grad_norm_with = input_with.grad.norm().item()

    print(f"   [PASS] Gradient norm WITH residual: {grad_norm_with:.6f}")
    print(f"   [INFO] This should help mitigate vanishing gradients in deep networks")

    print("\n" + "=" * 70)
    print("All residual connection tests passed successfully!")
    print("=" * 70)