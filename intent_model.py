import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPredictorHead(nn.Module):
    """
    A Transformer-based Temporal Prediction Head for Intent Prediction.
    Takes a sequence of feature embeddings and outputs a future action label.
    """
    def __init__(self, input_dim=2304, hidden_dim=512, num_classes=400, num_layers=2, nhead=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project high-dimensional concatenated features (e.g. 2304) to hidden_dim
        self.proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional Encoding (learnable) - assuming max sequence length ~ 64
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, hidden_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification head for future action label
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, input_dim]
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        B, T, _ = x.shape
        
        # Project
        x = self.proj(x) # [B, T, hidden_dim]
        
        # Add positional embedding
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer (batch_first=True expects [B, T, E])
        x = self.transformer(x) # [B, T, hidden_dim]
        
        # Pool across temporal dimension (e.g., take the mean or last token)
        # Here we use mean pooling over the sequence
        x = x.mean(dim=1) # [B, hidden_dim]
        
        # Classify
        logits = self.classifier(x) # [B, num_classes]
        return logits


class SlowFastIntentPredictor(nn.Module):
    """
    Wraps the pretrained SlowFast model to perform Intent Prediction.
    - Freezes the backbone.
    - Preserves the temporal dimension by bypassing the final temporal pooling.
    - Appends a Transformer-based TemporalPredictorHead.
    """
    def __init__(self, num_classes=400, freeze_backbone=True):
        super().__init__()
        
        # 1. Load the SlowFast ResNet-50 backbone from PyTorchVideo
        slowfast = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        
        # 2. Extract the backbone (everything before block 5 which does the temporal/spatial pooling)
        # The blocks are: 0-4 (ResNet stages)
        self.backbone = nn.ModuleList(slowfast.blocks[:5])
        
        # 3. Freeze backbone initially as requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 4. Initialize our custom Temporal Head
        # SlowFast R50 outputs 2048 channels for Fast and 256 for Slow (total 2304)
        self.temporal_head = TemporalPredictorHead(input_dim=2304, num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x: List of two tensors [slow_pathway, fast_pathway]
               slow_pathway shape: [B, C, T/alpha, H, W]
               fast_pathway shape: [B, C, T, H, W]
        Returns:
            logits: Future action predictions [B, num_classes]
        """
        # Pass through the SlowFast backbone stages
        for block in self.backbone:
            x = block(x)
            
        slow_out, fast_out = x # Shapes: [B, 2048, T_slow, H', W'], [B, 256, T_fast, H', W']
        
        # Spatially pool to [B, C, T, 1, 1] then squeeze to [B, C, T]
        slow_out = F.adaptive_avg_pool3d(slow_out, (slow_out.shape[2], 1, 1)).squeeze(-1).squeeze(-1)
        fast_out = F.adaptive_avg_pool3d(fast_out, (fast_out.shape[2], 1, 1)).squeeze(-1).squeeze(-1)
        
        # Transpose to [B, T, C] for sequence processing
        slow_out = slow_out.transpose(1, 2)
        fast_out = fast_out.transpose(1, 2)
        
        # Align temporal dimensions
        # The slow pathway has fewer frames (e.g. T/4). We interpolate it to match the fast pathway's sequence length.
        B, T_fast, C_fast = fast_out.shape
        slow_out = slow_out.transpose(1, 2) # Back to [B, C, T] for interpolation
        slow_out = F.interpolate(slow_out, size=T_fast, mode='linear', align_corners=False)
        slow_out = slow_out.transpose(1, 2) # [B, T_fast, C_slow]
        
        # Concatenate features along the channel dimension -> [B, T_fast, 2304]
        combined_features = torch.cat([slow_out, fast_out], dim=-1)
        
        # Pass the sequence through the Temporal Predictor Head
        logits = self.temporal_head(combined_features)
        
        return logits


if __name__ == "__main__":
    # --- Verification & Simulation ---
    print("--- Testing SlowFast Intent Predictor ---")
    
    # Instantiate the model
    model = SlowFastIntentPredictor(num_classes=10, freeze_backbone=True)
    model.eval()
    print("Model initialized and backbone frozen successfully.")
    
    # Check freezing
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    head_trainable = all(p.requires_grad for p in model.temporal_head.parameters())
    print(f"Backbone frozen: {backbone_frozen}")
    print(f"Temporal head trainable: {head_trainable}")
    
    # Create dummy tensors (simulating a 16-frame sequence)
    # Slow pathway is typically sampled at 1/4 the frame rate of Fast
    B = 2
    T_fast = 16
    T_slow = T_fast // 4
    H, W = 256, 256
    
    print(f"\nSimulating forward pass with {T_fast} frames (Batch Size: {B})...")
    # Slow pathway input shape: [B, C, T, H, W]
    dummy_slow = torch.randn(B, 3, T_slow, H, W)
    dummy_fast = torch.randn(B, 3, T_fast, H, W)
    
    # Forward pass
    with torch.no_grad():
        logits = model([dummy_slow, dummy_fast])
        
    print(f"Output logits shape: {logits.shape}") # Expected: [2, 10]
    print("Forward pass successful!")
    print("--- Test Complete ---")
