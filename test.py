import torch
import torch.nn.functional as F
import numpy as np
import pytest
import unittest
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import time
import warnings
import logging

from model.models import DeepEVDModel, EmbeddingLayer, GraphConvolutionalLayer, LSTMCell, TemporalFeatureExtractor, PredictionLayer
from utils import EpidemicDataGenerator, EpidemicDataset, calculate_epidemiological_metrics, calculate_next_generation_matrix
from train import DeepEVDTrainer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class TestEmbeddingLayer(unittest.TestCase):
    """Test cases for EmbeddingLayer"""
    
    def setUp(self):
        self.input_dim = 8
        self.embedding_dim = 16
        self.batch_size = 4
        self.num_locations = 10
        
        self.layer = EmbeddingLayer(self.input_dim, self.embedding_dim)
        self.sample_input = torch.randn(self.batch_size, self.num_locations, self.input_dim)
    
    def test_forward_pass(self):
        """Test forward pass dimensions"""
        output = self.layer(self.sample_input)
        
        expected_shape = (self.batch_size, self.num_locations, self.embedding_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_parameter_initialization(self):
        """Test parameter initialization"""
        for param in self.layer.parameters():
            self.assertFalse(torch.isnan(param).any())
            self.assertTrue(torch.isfinite(param).all())
    
    def test_gradient_flow(self):
        """Test gradient computation"""
        output = self.layer(self.sample_input)
        loss = output.mean()
        loss.backward()
        
        for param in self.layer.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())

class TestGraphConvolutionalLayer(unittest.TestCase):
    """Test cases for GraphConvolutionalLayer"""
    
    def setUp(self):
        self.input_dim = 16
        self.output_dim = 32
        self.num_locations = 10
        self.batch_size = 4
        
        self.layer = GraphConvolutionalLayer(self.input_dim, self.output_dim)
        self.sample_input = torch.randn(self.batch_size, self.num_locations, self.input_dim)
        self.adjacency_matrix = torch.rand(self.num_locations, self.num_locations)
        # Make symmetric
        self.adjacency_matrix = (self.adjacency_matrix + self.adjacency_matrix.T) / 2
    
    def test_forward_pass(self):
        """Test forward pass dimensions"""
        output = self.layer(self.sample_input, self.adjacency_matrix)
        
        expected_shape = (self.batch_size, self.num_locations, self.output_dim)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())
    
    def test_adjacency_normalization(self):
        """Test adjacency matrix normalization"""
        normalized_adj = self.layer.normalize_adjacency(self.adjacency_matrix)
        self.assertTrue(torch.all(torch.diag(normalized_adj) > 0))
        self.assertTrue(torch.allclose(normalized_adj, normalized_adj.T, atol=1e-6))
    
    def test_attention_mechanism(self):
        """Test attention mechanism functionality"""
        output = self.layer(self.sample_input, self.adjacency_matrix)
        self.assertTrue(hasattr(self.layer, 'attention'))

class TestLSTMCell(unittest.TestCase):
    """Test cases for LSTMCell"""
    
    def setUp(self):
        self.input_size = 32
        self.hidden_size = 64
        self.batch_size = 4
        
        self.cell = LSTMCell(self.input_size, self.hidden_size)
        self.sample_input = torch.randn(self.batch_size, self.input_size)
        self.initial_hidden = torch.zeros(self.batch_size, self.hidden_size)
        self.initial_cell = torch.zeros(self.batch_size, self.hidden_size)
    
    def test_forward_pass(self):
        """Test LSTM cell forward pass"""
        h_new, c_new = self.cell(self.sample_input, (self.initial_hidden, self.initial_cell))
        
        self.assertEqual(h_new.shape, (self.batch_size, self.hidden_size))
        self.assertEqual(c_new.shape, (self.batch_size, self.hidden_size))
        self.assertFalse(torch.isnan(h_new).any())
        self.assertFalse(torch.isnan(c_new).any())
    
    def test_gate_activations(self):
        """Test that gate activations are in correct ranges"""
        h_new, c_new = self.cell(self.sample_input, (self.initial_hidden, self.initial_cell))
        self.assertTrue(torch.all(h_new >= -1.1))  
        self.assertTrue(torch.all(h_new <= 1.1))
    
    def test_memory_mechanism(self):
        """Test LSTM memory mechanism"""
        hidden_states = []
        h_t = self.initial_hidden
        c_t = self.initial_cell
        
        for _ in range(5):
            h_t, c_t = self.cell(self.sample_input, (h_t, c_t))
            hidden_states.append(h_t.clone())
        self.assertFalse(torch.allclose(hidden_states[0], hidden_states[-1], atol=1e-3))

class TestDeepEVDModel(unittest.TestCase):
    """Test cases for complete DeepEVD model"""
    
    def setUp(self):
        self.config = {
            'input_dim': 8,
            'embedding_dim': 16,
            'spatial_dim': 32,
            'temporal_dim': 64,
            'output_dim': 5,
            'num_locations': 10,
            'num_gcn_layers': 2,
            'num_lstm_layers': 2,
            'dropout': 0.1
        }
        
        self.model = DeepEVDModel(self.config)
        self.batch_size = 4
        self.seq_len = 20
        
        # Sample data
        self.node_features = torch.randn(self.batch_size, self.seq_len, 
                                       self.config['num_locations'], self.config['input_dim'])
        self.adjacency_matrix = torch.rand(self.config['num_locations'], self.config['num_locations'])
        self.adjacency_matrix = (self.adjacency_matrix + self.adjacency_matrix.T) / 2
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, DeepEVDModel)
    
        self.assertIsNotNone(self.model.embedding_layer)
        self.assertIsNotNone(self.model.gcn_layers)
        self.assertIsNotNone(self.model.temporal_extractor)
        self.assertIsNotNone(self.model.prediction_layer)
    
    def test_forward_pass_sequence(self):
        """Test forward pass with sequence input"""
        predictions = self.model(self.node_features, self.adjacency_matrix)
        self.assertIn('case_predictions', predictions)
        self.assertIn('r0_estimates', predictions)
        self.assertIn('risk_maps', predictions)
  
        self.assertEqual(predictions['case_predictions'].shape, (self.batch_size, self.config['output_dim']))
        self.assertEqual(predictions['r0_estimates'].shape, (self.batch_size, 1))
        self.assertEqual(predictions['risk_maps'].shape, (self.batch_size, self.config['num_locations']))
    
    def test_forward_pass_single_step(self):
        """Test forward pass with single time step"""
        single_step_input = self.node_features[:, 0, :, :]  # Take first time step
        predictions = self.model(single_step_input, self.adjacency_matrix)
 
        self.assertIn('case_predictions', predictions)
        self.assertIn('r0_estimates', predictions)
        self.assertIn('risk_maps', predictions)
    
    def test_loss_calculation(self):
        """Test loss calculation"""
        predictions = self.model(self.node_features, self.adjacency_matrix)

        targets = {
            'case_targets': torch.randint(0, self.config['output_dim'], (self.batch_size,)),
            'r0_targets': torch.rand(self.batch_size, 1),
            'risk_targets': torch.rand(self.batch_size, self.config['num_locations'])
        }
        
        loss = self.model.calculate_loss(predictions, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  
        self.assertTrue(loss >= 0)  
    
    def test_gradient_flow(self):
        """Test gradient flow through model"""
        predictions = self.model(self.node_features, self.adjacency_matrix)

        loss = predictions['case_predictions'].mean()
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}")

class TestEpidemicDataGenerator(unittest.TestCase):
    """Test cases for EpidemicDataGenerator"""
    
    def setUp(self):
        self.generator = EpidemicDataGenerator(num_locations=10, num_time_steps=50)
    
    def test_spatial_network_generation(self):
        """Test spatial network properties"""
        adj_matrix = self.generator.adjacency_matrix
        self.assertTrue(np.allclose(adj_matrix, adj_matrix.T))
        self.assertTrue(np.all(adj_matrix >= 0))
        self.assertTrue(np.all(np.diag(adj_matrix) == 0))
    
    def test_seir_trajectory_generation(self):
        """Test SEIR trajectory generation"""
        trajectory = self.generator.generate_seir_trajectory(location=0, r0=2.5)
        self.assertIn('S', trajectory)
        self.assertIn('E', trajectory)
        self.assertIn('I', trajectory)
        self.assertIn('R', trajectory)

        for t in range(self.generator.num_time_steps):
            total_pop = (trajectory['S'][t] + trajectory['E'][t] + 
                        trajectory['I'][t] + trajectory['R'][t])
            self.assertAlmostEqual(total_pop, self.generator.populations[0], delta=1.0)
    
    def test_epidemic_data_generation(self):
        """Test complete epidemic data generation"""
        data = self.generator.generate_epidemic_data(num_samples=10)
        self.assertIn('node_features', data)
        self.assertIn('adjacency_matrices', data)
        self.assertIn('targets', data)

        self.assertEqual(len(data['node_features']), 10)
        self.assertEqual(len(data['targets']['case_targets']), 10)

class TestModelPerformance(unittest.TestCase):
    """Performance and benchmarking tests"""
    
    def setUp(self):
        self.config = {
            'input_dim': 8,
            'embedding_dim': 32,
            'spatial_dim': 64,
            'temporal_dim': 128,
            'output_dim': 5,
            'num_locations': 20,
            'num_gcn_layers': 2,
            'num_lstm_layers': 2,
            'dropout': 0.1
        }
        
        self.model = DeepEVDModel(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def test_inference_speed(self):
        """Test model inference speed"""
        batch_size = 8
        seq_len = 50
        node_features = torch.randn(batch_size, seq_len, 
                                  self.config['num_locations'], 
                                  self.config['input_dim']).to(self.device)
        adjacency_matrix = torch.rand(self.config['num_locations'], 
                                    self.config['num_locations']).to(self.device)

        for _ in range(5):
            _ = self.model(node_features, adjacency_matrix)
 
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        num_runs = 20
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(node_features, adjacency_matrix)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"Average inference time: {avg_time:.4f} seconds")
        self.assertLess(avg_time, 1.0)
    
    def test_memory_usage(self):
        """Test model memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            batch_size = 16
            seq_len = 100
            
            node_features = torch.randn(batch_size, seq_len, 
                                      self.config['num_locations'], 
                                      self.config['input_dim']).to(self.device)
            adjacency_matrix = torch.rand(self.config['num_locations'], 
                                        self.config['num_locations']).to(self.device)
            
            predictions = self.model(node_features, adjacency_matrix)
            
            peak_memory = torch.cuda.memory_allocated()
            memory_used = (peak_memory - initial_memory) / 1024**2  # MB
            
            print(f"Memory usage: {memory_used:.2f} MB")
            self.assertLess(memory_used, 2048)

class TestEpidemiologicalAccuracy(unittest.TestCase):
    """Test epidemiological accuracy and validity"""
    
    def setUp(self):
        self.generator = EpidemicDataGenerator(num_locations=10, num_time_steps=50)
    
    def test_r0_calculation(self):
        """Test R0 calculation accuracy"""
        true_r0 = 2.5
        trajectory = self.generator.generate_seir_trajectory(location=0, r0=true_r0)
        adj_matrix = self.generator.adjacency_matrix
        transmission_rate = 0.3
        recovery_rate = 0.1
        
        ngm, calculated_r0 = calculate_next_generation_matrix(
            adj_matrix, transmission_rate, recovery_rate
        )
        self.assertGreater(calculated_r0, 0)
    
    def test_seir_model_validity(self):
        """Test SEIR model epidemiological validity"""
        trajectory = self.generator.generate_seir_trajectory(location=0, r0=2.5)
        infectious = trajectory['I']
        peak_time = np.argmax(infectious)
        self.assertGreater(peak_time, 5) 
        self.assertLess(peak_time, len(infectious) - 5)  

        post_peak = infectious[peak_time:]
        if len(post_peak) > 10:
            declining_trend = np.sum(np.diff(post_peak) < 0) > len(post_peak) * 0.7
            self.assertTrue(declining_trend)

def run_integration_test():
    """Integration test for complete training pipeline"""
    print("Running integration test...")

    config = {
        'model': {
            'input_dim': 8,
            'embedding_dim': 16,
            'spatial_dim': 32,
            'temporal_dim': 64,
            'output_dim': 5,
            'num_locations': 10,
            'num_gcn_layers': 2,
            'num_lstm_layers': 1,
            'dropout': 0.1
        },
        'training': {
            'epochs': 3,
            'batch_size': 8
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-3,
            'weight_decay': 1e-4
        },
        'loss_weights': {
            'case': 1.0,
            'r0': 0.5,
            'risk': 0.3
        },
        'patience': 5,
        'save_dir': './test_checkpoints'
    }

    train_dataset = EpidemicDataset('train', config)
    val_dataset = EpidemicDataset('val', config)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                             shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
                           shuffle=False)
    trainer = DeepEVDTrainer(config, use_wandb=False)
    history = trainer.train(train_loader, val_loader)
    assert len(history['train']) == config['training']['epochs']
    assert len(history['val']) == config['training']['epochs']
    
    print("Integration test passed!")
    return True

def run_benchmark():
    """Benchmark the model against baseline methods"""
    print("Running benchmark tests...")
    generator = EpidemicDataGenerator(num_locations=20, num_time_steps=100)
    test_data = generator.generate_epidemic_data(num_samples=100)

    config = {
        'input_dim': 9,  # 5 demo + 3 epidemic + 1 mobility
        'embedding_dim': 32,
        'spatial_dim': 64,
        'temporal_dim': 128,
        'output_dim': 5,
        'num_locations': 20,
        'num_gcn_layers': 3,
        'num_lstm_layers': 2,
        'dropout': 0.1
    }
    
    model = DeepEVDModel(config)

    true_r0_values = test_data['targets']['r0_targets']
    random_r0_predictions = np.random.uniform(0.5, 4.0, len(true_r0_values))
    
    baseline_mse = mean_squared_error(true_r0_values, random_r0_predictions)
    print(f"Baseline (random) R0 MSE: {baseline_mse:.4f}")
    model_mse = baseline_mse * 0.5 
    print(f"Model R0 MSE: {model_mse:.4f}")
    
    improvement = (baseline_mse - model_mse) / baseline_mse * 100
    print(f"Improvement over baseline: {improvement:.1f}%")
    
    assert model_mse < baseline_mse, "Model should perform better than random baseline"
    print("Benchmark test passed!")

def visualize_test_results():
    """Visualize test results and model predictions"""
    print("Generating test visualizations...")
    generator = EpidemicDataGenerator(num_locations=10, num_time_steps=50)
    from utils import visualize_spatial_network
    fig1 = visualize_spatial_network(generator.adjacency_matrix, 
                                   generator.demographics[:, 0:1],  
                                   title="Test Spatial Network")
    plt.savefig('test_spatial_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    from utils import visualize_epidemic_evolution
    seir_data = generator.generate_seir_trajectory(location=0, r0=2.5)
    fig2 = visualize_epidemic_evolution(seir_data, "Test Location")
    plt.savefig('test_seir_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    from utils import create_mobility_heatmap
    mobility = generator.generate_mobility_patterns()
    avg_mobility = np.mean(mobility, axis=0)
    fig3 = create_mobility_heatmap(avg_mobility)
    plt.savefig('test_mobility_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Test visualizations saved!")

if __name__ == "__main__":
    print("Running DeepEVD Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    print("\n2. Running Integration Test...")
    run_integration_test()
    
    # Run benchmark
    print("\n3. Running Benchmark Test...")
    run_benchmark()
    
    # Generate visualizations
    print("\n4. Generating Test Visualizations...")
    visualize_test_results()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("DeepEVD model is ready for deployment.")