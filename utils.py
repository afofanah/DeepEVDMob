import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import networkx as nx
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import pickle

def setup_logging(level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deepevd.log'),
            logging.StreamHandler()
        ]
    )

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

class EpidemicDataGenerator:
    """
    Generate synthetic epidemic data for training and testing
    Simulates realistic spatio-temporal disease transmission patterns
    """
    def __init__(self, num_locations: int = 20, num_time_steps: int = 100, 
                 seed: int = 42):
        self.num_locations = num_locations
        self.num_time_steps = num_time_steps
        self.seed = seed
        np.random.seed(seed)
        
        # Generate spatial network
        self.adjacency_matrix = self._generate_spatial_network()
        self.distance_matrix = self._generate_distance_matrix()
        
        # Population and demographic features
        self.populations = np.random.lognormal(10, 1, num_locations)
        self.demographics = self._generate_demographics()
        
    def _generate_spatial_network(self) -> np.ndarray:
        """Generate realistic spatial network based on geographical principles"""
        G = nx.barabasi_albert_graph(self.num_locations, 3, seed=self.seed)
        
        # Add geographical constraints
        positions = np.random.rand(self.num_locations, 2) * 100 
        
        adj_matrix = np.zeros((self.num_locations, self.num_locations))
        
        for i in range(self.num_locations):
            for j in range(i+1, self.num_locations):
                distance = np.linalg.norm(positions[i] - positions[j])
        
                prob = np.exp(-distance / 20)  
                if np.random.rand() < prob or G.has_edge(i, j):
                    weight = (self.populations[i] * self.populations[j]) / (distance + 1)
                    adj_matrix[i, j] = adj_matrix[j, i] = weight
        
        return adj_matrix
    
    def _generate_distance_matrix(self) -> np.ndarray:
        """Generate distance matrix for locations"""
        positions = np.random.rand(self.num_locations, 2) * 100
        return squareform(pdist(positions, metric='euclidean'))
    
    def _generate_demographics(self) -> np.ndarray:
        """Generate demographic features for each location"""
        features = []
        
        for i in range(self.num_locations):
            mean_age = np.random.normal(35, 10)
            density = self.populations[i] / (np.random.uniform(1, 100))
            income = np.random.lognormal(10, 0.5)
            education = np.random.beta(2, 5)  
            healthcare_capacity = np.random.gamma(2, 2)
            
            features.append([mean_age, density, income, education, healthcare_capacity])
        
        return np.array(features)
    
    def generate_seir_trajectory(self, location: int, r0: float = 2.5, 
                                initial_infected: int = 1) -> Dict[str, np.ndarray]:
        """Generate SEIR trajectory for a single location"""
        population = int(self.populations[location])
        incubation_period = 5.1  # days
        infectious_period = 3.0  # days
        sigma = 1 / incubation_period  # Incubation rate
        gamma = 1 / infectious_period  # Recovery rate
        beta = r0 * gamma / population  # Transmission rate
        
        S = np.zeros(self.num_time_steps)
        E = np.zeros(self.num_time_steps)
        I = np.zeros(self.num_time_steps)
        R = np.zeros(self.num_time_steps)
        
        S[0] = population - initial_infected
        E[0] = 0
        I[0] = initial_infected
        R[0] = 0
        
        # Simulate SEIR dynamics
        for t in range(1, self.num_time_steps):
            dS = -beta * S[t-1] * I[t-1]
            dE = beta * S[t-1] * I[t-1] - sigma * E[t-1]
            dI = sigma * E[t-1] - gamma * I[t-1]
            dR = gamma * I[t-1]
            
            S[t] = max(0, S[t-1] + dS)
            E[t] = max(0, E[t-1] + dE)
            I[t] = max(0, I[t-1] + dI)
            R[t] = max(0, R[t-1] + dR)
        
        return {'S': S, 'E': E, 'I': I, 'R': R}
    
    def generate_mobility_patterns(self) -> np.ndarray:
        """Generate realistic mobility patterns between locations"""
        mobility = np.zeros((self.num_time_steps, self.num_locations, self.num_locations))
        
        for t in range(self.num_time_steps):
            base_mobility = self.adjacency_matrix.copy()
            day_of_week = t % 7
            hour_of_day = (t * 24 / self.num_time_steps) % 24
            if day_of_week >= 5:
                base_mobility *= 0.7
            if hour_of_day < 6 or hour_of_day > 22:
                base_mobility *= 0.3

            noise = np.random.normal(1, 0.1, base_mobility.shape)
            base_mobility *= noise
            
            mobility[t] = np.maximum(0, base_mobility)
        
        return mobility
    
    def generate_epidemic_data(self, num_samples: int = 1000) -> Dict:
        """Generate comprehensive epidemic dataset"""
        data = {
            'node_features': [],
            'adjacency_matrices': [],
            'temporal_sequences': [],
            'targets': {
                'case_targets': [],
                'r0_targets': [],
                'risk_targets': []
            },
            'metadata': {
                'populations': self.populations,
                'demographics': self.demographics,
                'distance_matrix': self.distance_matrix
            }
        }
        
        for sample in range(num_samples):
            r0 = np.random.uniform(0.8, 4.0)
            seir_data = {}
            for loc in range(self.num_locations):
                initial_infected = np.random.poisson(2) + 1
                seir_data[loc] = self.generate_seir_trajectory(loc, r0, initial_infected)
            
            mobility = self.generate_mobility_patterns()
            node_features = []
            for t in range(self.num_time_steps):
                time_features = []
                for loc in range(self.num_locations):
                    demo_features = self.demographics[loc]
   
                    current_I = seir_data[loc]['I'][t]
                    current_R = seir_data[loc]['R'][t]
                    current_S = seir_data[loc]['S'][t]
                    mobility_out = np.sum(mobility[t, loc, :])
                    
                    features = np.concatenate([
                        demo_features,  
                        [current_I, current_R, current_S],  
                        [mobility_out]  
                    ]) 
                    time_features.append(features)
                
                node_features.append(time_features)

            case_target = self._discretize_cases([seir_data[loc]['I'][-1] for loc in range(self.num_locations)])
            r0_target = r0
            risk_target = self._calculate_risk_scores(seir_data, mobility)
            
            # Store data
            data['node_features'].append(np.array(node_features))
            data['adjacency_matrices'].append(self.adjacency_matrix)
            data['temporal_sequences'].append(mobility)
            data['targets']['case_targets'].append(case_target)
            data['targets']['r0_targets'].append(r0_target)
            data['targets']['risk_targets'].append(risk_target)
        
        return data
    
    def _discretize_cases(self, case_counts: List[float], num_classes: int = 5) -> int:
        """Discretize continuous case counts into classes"""
        max_cases = max(case_counts)
        if max_cases == 0:
            return 0
        bins = np.logspace(0, np.log10(max_cases + 1), num_classes + 1)
        total_cases = sum(case_counts)
        
        for i, bin_edge in enumerate(bins[1:]):
            if total_cases <= bin_edge:
                return i
        
        return num_classes - 1
    
    def _calculate_risk_scores(self, seir_data: Dict, mobility: np.ndarray) -> np.ndarray:
        """Calculate risk scores for each location"""
        risk_scores = np.zeros(self.num_locations)
        
        for loc in range(self.num_locations):
            current_risk = seir_data[loc]['I'][-1] / self.populations[loc]
            connectivity_risk = 0
            for other_loc in range(self.num_locations):
                if other_loc != loc:
                    other_risk = seir_data[other_loc]['I'][-1] / self.populations[other_loc]
                    mobility_strength = np.mean(mobility[:, loc, other_loc])
                    connectivity_risk += other_risk * mobility_strength

            total_risk = current_risk + 0.3 * connectivity_risk
            risk_scores[loc] = min(1.0, total_risk)
        
        return risk_scores

class EpidemicDataset(Dataset):
    """
    PyTorch Dataset for epidemic data
    Handles loading and preprocessing of epidemic datasets
    """
    def __init__(self, split: str, config: Dict, data_path: Optional[str] = None):
        self.split = split
        self.config = config
        self.data_path = data_path
        
        if data_path is None:
            self.generator = EpidemicDataGenerator(
                num_locations=config['model']['num_locations'],
                num_time_steps=config.get('num_time_steps', 50)
            )
            self.data = self._generate_synthetic_data()
        else:
            self.data = self._load_real_data(data_path)

        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        self._preprocess_data()
    
    def _generate_synthetic_data(self) -> Dict:
        """Generate synthetic data for the specified split"""
        num_samples = 1000 if self.split == 'train' else 200
        return self.generator.generate_epidemic_data(num_samples)
    
    def _load_real_data(self, data_path: str) -> Dict:
        """Load real epidemic data from file"""
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    
    def _preprocess_data(self):
        """Preprocess the data (normalization, etc.)"""
        all_features = []
        for sample_features in self.data['node_features']:
            for time_step in sample_features:
                all_features.extend(time_step)

        self.feature_scaler.fit(all_features)
        for i, sample_features in enumerate(self.data['node_features']):
            normalized_sample = []
            for time_step in sample_features:
                normalized_step = self.feature_scaler.transform(time_step)
                normalized_sample.append(normalized_step)
            self.data['node_features'][i] = np.array(normalized_sample)
        
        r0_targets = np.array(self.data['targets']['r0_targets']).reshape(-1, 1)
        self.target_scaler.fit(r0_targets)
        normalized_r0 = self.target_scaler.transform(r0_targets).flatten()
        self.data['targets']['r0_targets'] = normalized_r0.tolist()
    
    def __len__(self) -> int:
        return len(self.data['node_features'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get sample data
        node_features = torch.FloatTensor(self.data['node_features'][idx])
        adjacency_matrix = torch.FloatTensor(self.data['adjacency_matrices'][idx])
        
        # Get targets
        case_target = torch.LongTensor([self.data['targets']['case_targets'][idx]])
        r0_target = torch.FloatTensor([self.data['targets']['r0_targets'][idx]])
        risk_target = torch.FloatTensor(self.data['targets']['risk_targets'][idx])
        
        return {
            'node_features': node_features,
            'adjacency_matrix': adjacency_matrix,
            'targets': {
                'case_targets': case_target.squeeze(),
                'r0_targets': r0_target.squeeze(),
                'risk_targets': risk_target
            }
        }

def calculate_epidemiological_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    Calculate epidemiological evaluation metrics
    """
    metrics = {}
    
    # Aggregate all predictions and targets
    all_r0_pred = []
    all_r0_true = []
    all_risk_pred = []
    all_risk_true = []
    
    for pred_batch, target_batch in zip(predictions, targets):
        if 'r0_estimates' in pred_batch and 'r0_targets' in target_batch:
            all_r0_pred.extend(pred_batch['r0_estimates'].numpy().flatten())
            all_r0_true.extend(target_batch['r0_targets'].numpy().flatten())
        
        if 'risk_maps' in pred_batch and 'risk_targets' in target_batch:
            all_risk_pred.extend(pred_batch['risk_maps'].numpy().flatten())
            all_risk_true.extend(target_batch['risk_targets'].numpy().flatten())
    
    # R0 estimation metrics
    if all_r0_pred and all_r0_true:
        metrics['r0_mae'] = mean_absolute_error(all_r0_true, all_r0_pred)
        metrics['r0_mse'] = mean_squared_error(all_r0_true, all_r0_pred)
        metrics['r0_r2'] = r2_score(all_r0_true, all_r0_pred)
        
        # R0 threshold accuracy (R0 > 1 indicates outbreak)
        r0_true_binary = np.array(all_r0_true) > 1
        r0_pred_binary = np.array(all_r0_pred) > 1
        metrics['r0_threshold_accuracy'] = np.mean(r0_true_binary == r0_pred_binary)
    
    # Risk prediction metrics
    if all_risk_pred and all_risk_true:
        metrics['risk_mae'] = mean_absolute_error(all_risk_true, all_risk_pred)
        metrics['risk_mse'] = mean_squared_error(all_risk_true, all_risk_pred)
        
        # Correlation between predicted and true risk
        if len(set(all_risk_true)) > 1:  # Check for variance
            correlation, _ = pearsonr(all_risk_true, all_risk_pred)
            metrics['risk_correlation'] = correlation
    
    return metrics

def calculate_next_generation_matrix(adjacency: np.ndarray, transmission_rate: float, 
                                   recovery_rate: float) -> Tuple[np.ndarray, float]:
    """
    Calculate Next Generation Matrix and basic reproduction number R0
    """
    num_locations = adjacency.shape[0]
    F = transmission_rate * adjacency
    V = recovery_rate * np.eye(num_locations)
    try:
        V_inv = np.linalg.inv(V)
        NGM = np.dot(F, V_inv)
        eigenvalues = np.linalg.eigvals(NGM)
        R0 = np.real(np.max(eigenvalues))
        
        return NGM, R0
    except np.linalg.LinAlgError:
        return np.zeros_like(F), 0.0

def visualize_spatial_network(adjacency_matrix: np.ndarray, 
                            node_features: Optional[np.ndarray] = None,
                            title: str = "Spatial Network") -> plt.Figure:
    """
    Visualize spatial network with optional node features
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G, k=1, iterations=50)
    if node_features is not None:
        node_colors = node_features[:, 0]  
        cmap = plt.cm.Reds
    else:
        node_colors = 'lightblue'
        cmap = None
    node_sizes = [G.degree(node) * 100 + 200 for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          cmap=cmap, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    if node_features is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=node_colors.min(), 
                                                                vmax=node_colors.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Feature Value')
    
    plt.tight_layout()
    return fig

def visualize_epidemic_evolution(seir_data: Dict[str, np.ndarray], 
                               location_name: str = "Location") -> plt.Figure:
    """
    Visualize SEIR epidemic evolution over time
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    time_steps = len(seir_data['S'])
    time_points = range(time_steps)
    
    ax.plot(time_points, seir_data['S'], 'b-', label='Susceptible', linewidth=2)
    ax.plot(time_points, seir_data['E'], 'orange', label='Exposed', linewidth=2)
    ax.plot(time_points, seir_data['I'], 'r-', label='Infectious', linewidth=2)
    ax.plot(time_points, seir_data['R'], 'g-', label='Recovered', linewidth=2)
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Population')
    ax.set_title(f'SEIR Model Evolution - {location_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_mobility_heatmap(mobility_matrix: np.ndarray, 
                          location_names: Optional[List[str]] = None) -> plt.Figure:
    """
    Create heatmap visualization of mobility patterns
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if location_names is None:
        location_names = [f'Loc_{i}' for i in range(mobility_matrix.shape[0])]
    
    sns.heatmap(mobility_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=location_names, yticklabels=location_names, ax=ax)
    
    ax.set_title('Mobility Patterns Between Locations')
    ax.set_xlabel('Destination')
    ax.set_ylabel('Origin')
    
    plt.tight_layout()
    return fig

def analyze_model_attention(model, sample_data: Dict) -> Dict[str, np.ndarray]:
    """
    Analyze attention weights in the trained model
    """
    model.eval()
    
    with torch.no_grad():
        predictions = model(sample_data['node_features'], sample_data['adjacency_matrix'])
        
        attention_weights = {}
        if 'spatial_attention' in predictions:
            attention_weights['spatial'] = predictions['spatial_attention'].numpy()  
        if 'temporal_attention' in predictions:
            attention_weights['temporal'] = predictions['temporal_attention'].numpy()
    
    return attention_weights

def export_results_to_csv(results: Dict, output_path: str):
    df_data = []
    
    for key, values in results.items():
        if isinstance(values, (list, np.ndarray)):
            for i, value in enumerate(values):
                df_data.append({'metric': key, 'index': i, 'value': value})
        else:
            df_data.append({'metric': key, 'index': 0, 'value': values})
    
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    
    print(f"Results exported to {output_path}")

def calculate_model_complexity(model) -> Dict[str, int]:
    """
    Calculate model complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    component_params = {}
    for name, module in model.named_children():
        component_params[name] = sum(p.numel() for p in module.parameters())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'component_parameters': component_params
    }

def validate_data_quality(dataset: EpidemicDataset) -> Dict[str, Union[bool, str]]:
    """
    Validate the quality of epidemic dataset
    """
    validation_results = {
        'valid': True,
        'issues': []
    }
    for i, sample in enumerate(dataset.data['node_features']):
        if np.isnan(sample).any():
            validation_results['valid'] = False
            validation_results['issues'].append(f"NaN values in sample {i}")

    for i, adj in enumerate(dataset.data['adjacency_matrices']):
        if not np.allclose(adj, adj.T):
            validation_results['valid'] = False
            validation_results['issues'].append(f"Non-symmetric adjacency matrix in sample {i}")
        
        if np.any(adj < 0):
            validation_results['valid'] = False
            validation_results['issues'].append(f"Negative values in adjacency matrix in sample {i}")
    
    r0_targets = dataset.data['targets']['r0_targets']
    if any(r0 < 0 for r0 in r0_targets):
        validation_results['valid'] = False
        validation_results['issues'].append("Negative R0 values found")
    
    risk_targets = dataset.data['targets']['risk_targets']
    for i, risk in enumerate(risk_targets):
        if np.any(risk < 0) or np.any(risk > 1):
            validation_results['valid'] = False
            validation_results['issues'].append(f"Risk values out of [0,1] range in sample {i}")
    
    return validation_results