import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModalityExpert(nn.Module):
    """Expert network with convolution for processing a single modality with variable sequence lengths."""
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3):
        super(ConvModalityExpert, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=2, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Convolution for feature extraction
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # Adaptive pooling to align feature sizes
        x = self.dropout(x)
        x = self.fc(x)
        return x

class AttentionGating(nn.Module):
    """Attention-based gating network for deciding modality importance."""
    def __init__(self, num_modalities, hidden_dim):
        super(AttentionGating, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)
        self.fc = nn.Linear(hidden_dim, num_modalities)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, modality_features):
        modality_features = torch.stack(modality_features, dim=0)
        attention_output, _ = self.attention(modality_features, modality_features, modality_features)
        combined = torch.mean(attention_output, dim=0)
        weights = self.softmax(self.fc(combined))
        return weights

class TemporalModalityAwareExpert(nn.Module):
    """Combines modality-specific experts and attention-based gating for variable-length modalities."""
    def __init__(self, input_dims, hidden_dim, output_dim):
        super(TemporalModalityAwareExpert, self).__init__()
        self.num_modalities = len(input_dims)

        # Create expert networks for each modality
        self.experts = nn.ModuleList([
            ConvModalityExpert(input_dim, hidden_dim, output_dim)
            for input_dim in input_dims
        ])

        # Create attention-based gating network
        self.gating_network = AttentionGating(self.num_modalities, hidden_dim)

    def forward(self, *modality_inputs):
        batch_size = modality_inputs[0].shape[0]

        # Process each modality independently
        modality_features = [
            self.experts[i](modality_inputs[i].permute(0, 2, 1))  # Permute for Conv1d processing
            for i in range(self.num_modalities)
        ]

        # Compute gating weights
        gating_weights = self.gating_network(modality_features)

        # Weighted combination of modality features
        weighted_features = torch.stack(modality_features, dim=0) * gating_weights.unsqueeze(-1)
        final_output = weighted_features.sum(dim=0)

        return final_output, gating_weights
