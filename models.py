import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List

class EmbeddingLayer(nn.Module):
    """
    Embedding layer for location co-occurrence matrix factorization
    Creates dense vector representations from sparse mobility data
    """
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1):
        super(EmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Location embeddings
        self.location_embedding = nn.Linear(input_dim, embedding_dim)
        self.context_embedding = nn.Linear(input_dim, embedding_dim)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.location_embedding.weight)
        nn.init.xavier_uniform_(self.context_embedding.weight)
        nn.init.zeros_(self.location_embedding.bias)
        nn.init.zeros_(self.context_embedding.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loc_emb = self.location_embedding(x)
        ctx_emb = self.context_embedding(x)
        combined = loc_emb * ctx_emb + loc_emb + ctx_emb
        embedded = self.layer_norm(combined)
        embedded = self.dropout(embedded)
        
        return embedded

class GraphConvolutionalLayer(nn.Module):
    """
    Graph Convolutional Network layer for spatial feature extraction
    Implements the SFRE module from the paper
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super(GraphConvolutionalLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4, dropout=dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
  
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
      
        degree_matrix = torch.diag(degree_inv_sqrt)
        
        # Symmetric normalization
        adj_norm = torch.mm(torch.mm(degree_matrix, adj), degree_matrix)
        
        return adj_norm
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape

        x_reshaped = x.transpose(0, 1)  
        attn_output, attn_weights = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_combined = x + attn_output.transpose(0, 1)
        adj_norm = self.normalize_adjacency(adj)
        output = torch.zeros(batch_size, num_nodes, self.output_dim, device=x.device)
        
        for b in range(batch_size):
            support = torch.mm(x_combined[b], self.weight)
            output[b] = torch.mm(adj_norm, support) + self.bias
        
        output = F.relu(output)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output

class LSTMCell(nn.Module):
    """
    Custom LSTM cell for temporal feature extraction
    Implements the TFRE module with enhanced memory mechanisms
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM gates
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_c = nn.LayerNorm(hidden_size)
        self.layer_norm_h = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1
                if 'forget_gate' in name:
                    param.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(self, x: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = hidden_state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        i_t = torch.sigmoid(self.input_gate(combined))      
        f_t = torch.sigmoid(self.forget_gate(combined))     
        o_t = torch.sigmoid(self.output_gate(combined))     
        g_t = torch.tanh(self.cell_gate(combined))          
        
        # Update cell state
        c_t = f_t * c_prev + i_t * g_t
        c_t = self.layer_norm_c(c_t)
        
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        h_t = self.layer_norm_h(h_t)
        h_t = self.dropout(h_t)
        
        return h_t, c_t

class TemporalFeatureExtractor(nn.Module):
    """
    Temporal Feature Representation and Extraction (TFRE) module
    Uses LSTM with dilated convolutions for multi-scale temporal patterns
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super(TemporalFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, dropout)
            for i in range(num_layers)
        ])
        
        self.conv1d_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2**i, padding=2**i)
            for i in range(3)
        ])
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.output_proj = nn.Linear(hidden_dim * 4, hidden_dim)  
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hidden_states: Optional[List] = None) -> Tuple[torch.Tensor, List]:
        batch_size, seq_len, _ = x.shape
        
        if hidden_states is None:
            hidden_states = [
                (torch.zeros(batch_size, self.hidden_dim, device=x.device),
                 torch.zeros(batch_size, self.hidden_dim, device=x.device))
                for _ in range(self.num_layers)
            ]
        lstm_outputs = []
        new_hidden_states = []
        
        for t in range(seq_len):
            layer_input = x[:, t, :]
            
            for i, lstm_cell in enumerate(self.lstm_cells):
                h_t, c_t = lstm_cell(layer_input, hidden_states[i])
                hidden_states[i] = (h_t, c_t)
                layer_input = h_t
            
            lstm_outputs.append(h_t)
            new_hidden_states.append(hidden_states.copy())

        lstm_output = torch.stack(lstm_outputs, dim=1)  
        conv_features = []
        lstm_transposed = lstm_output.transpose(1, 2)  
        
        for conv_layer in self.conv1d_layers:
            conv_out = F.relu(conv_layer(lstm_transposed))
            conv_features.append(conv_out.transpose(1, 2))  
        
        lstm_permuted = lstm_output.transpose(0, 1)  
        attn_output, _ = self.temporal_attention(lstm_permuted, lstm_permuted, lstm_permuted)
        attn_output = attn_output.transpose(0, 1)  
        combined_features = torch.cat([lstm_output] + conv_features, dim=-1)
        
        # Final projection
        temporal_features = self.output_proj(combined_features)
        temporal_features = self.dropout(temporal_features)
        
        return temporal_features, new_hidden_states[-1]

class PredictionLayer(nn.Module):
    """
    Prediction layer that combines spatial and temporal features
    Generates multiple outputs: case predictions, R0 estimates, and risk maps
    """
    def __init__(self, spatial_dim: int, temporal_dim: int, output_dim: int, num_locations: int):
        super(PredictionLayer, self).__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.output_dim = output_dim
        self.num_locations = num_locations
        
        combined_dim = spatial_dim + temporal_dim
        
        # Case prediction head
        self.case_predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, output_dim)
        )
        
        # R0 estimation head
        self.r0_predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 4),
            nn.ReLU(),
            nn.Linear(combined_dim // 4, 1),
            nn.Softplus()  # Ensure R0 > 0
        )
        
        # Risk mapping head
        self.risk_predictor = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Linear(combined_dim // 2, num_locations),
            nn.Sigmoid()  # Risk scores between 0 and 1
        )
        
        # Attention weights for feature fusion
        self.spatial_attention = nn.Linear(spatial_dim, 1)
        self.temporal_attention = nn.Linear(temporal_dim, 1)
    
    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = spatial_features.shape[0]
        spatial_attn_weights = F.softmax(self.spatial_attention(spatial_features), dim=1)
        spatial_agg = torch.sum(spatial_features * spatial_attn_weights, dim=1)

        temporal_attn_weights = F.softmax(self.temporal_attention(temporal_features), dim=1)
        temporal_agg = torch.sum(temporal_features * temporal_attn_weights, dim=1)

        combined_features = torch.cat([spatial_agg, temporal_agg], dim=-1)
        
        case_predictions = self.case_predictor(combined_features)
        r0_estimates = self.r0_predictor(combined_features)
        risk_maps = self.risk_predictor(combined_features)
        
        return {
            'case_predictions': F.softmax(case_predictions, dim=-1),
            'r0_estimates': r0_estimates,
            'risk_maps': risk_maps,
            'spatial_attention': spatial_attn_weights,
            'temporal_attention': temporal_attn_weights
        }

class DeepEVDModel(nn.Module):
    """
    Complete DeepEVD model for spatio-temporal disease transmission prediction
    Integrates embedding, spatial GCN, temporal LSTM, and prediction components
    """
    def __init__(self, config: Dict):
        super(DeepEVDModel, self).__init__()
        self.config = config
        
        # Extract configuration
        input_dim = config['input_dim']
        embedding_dim = config['embedding_dim']
        spatial_dim = config['spatial_dim']
        temporal_dim = config['temporal_dim']
        output_dim = config['output_dim']
        num_locations = config['num_locations']
        num_gcn_layers = config.get('num_gcn_layers', 2)
        num_lstm_layers = config.get('num_lstm_layers', 2)
        dropout = config.get('dropout', 0.1)
   
        self.embedding_layer = EmbeddingLayer(input_dim, embedding_dim, dropout)
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim = embedding_dim if i == 0 else spatial_dim
            self.gcn_layers.append(GraphConvolutionalLayer(in_dim, spatial_dim, dropout))

        self.temporal_extractor = TemporalFeatureExtractor(
            spatial_dim, temporal_dim, num_lstm_layers, dropout
        )
        self.prediction_layer = PredictionLayer(
            spatial_dim, temporal_dim, output_dim, num_locations
        )
        self.hidden_states = None
    
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor, 
                sequence_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if len(node_features.shape) == 4:  
            batch_size, seq_len, num_locations, input_dim = node_features.shape
            spatial_outputs = []
            for t in range(seq_len):
                embedded = self.embedding_layer(node_features[:, t])
                spatial_features = embedded
                for gcn_layer in self.gcn_layers:
                    spatial_features = gcn_layer(spatial_features, adjacency_matrix)
                spatial_agg = torch.mean(spatial_features, dim=1) 
                spatial_outputs.append(spatial_agg)
 
            temporal_input = torch.stack(spatial_outputs, dim=1)
            temporal_features, self.hidden_states = self.temporal_extractor(
                temporal_input, self.hidden_states
            )
            final_spatial = spatial_features  
            final_temporal = temporal_features[:, -1:, :]  
            
        else: 
            batch_size, num_locations, input_dim = node_features.shape
            embedded = self.embedding_layer(node_features)
            spatial_features = embedded
            for gcn_layer in self.gcn_layers:
                spatial_features = gcn_layer(spatial_features, adjacency_matrix)
            
            spatial_agg = torch.mean(spatial_features, dim=1, keepdim=True)
            temporal_features, self.hidden_states = self.temporal_extractor(
                spatial_agg, self.hidden_states
            )
            
            final_spatial = spatial_features
            final_temporal = temporal_features
        predictions = self.prediction_layer(final_spatial, final_temporal)
        
        return predictions
    
    def reset_hidden_states(self):
        """Reset hidden states for temporal processing"""
        self.hidden_states = None
    
    def calculate_loss(self, predictions: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor], 
                      loss_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        if loss_weights is None:
            loss_weights = {'case': 1.0, 'r0': 0.5, 'risk': 0.3}
        
        total_loss = 0.0
        if 'case_predictions' in predictions and 'case_targets' in targets:
            case_loss = F.cross_entropy(predictions['case_predictions'], targets['case_targets'])
            total_loss += loss_weights['case'] * case_loss
 
        if 'r0_estimates' in predictions and 'r0_targets' in targets:
            r0_loss = F.mse_loss(predictions['r0_estimates'], targets['r0_targets'])
            total_loss += loss_weights['r0'] * r0_loss
        
        if 'risk_maps' in predictions and 'risk_targets' in targets:
            risk_loss = F.binary_cross_entropy(predictions['risk_maps'], targets['risk_targets'])
            total_loss += loss_weights['risk'] * risk_loss
        
        return total_loss