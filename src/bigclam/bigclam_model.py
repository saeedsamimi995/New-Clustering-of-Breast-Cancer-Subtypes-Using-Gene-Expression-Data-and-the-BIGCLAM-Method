"""BIGCLAM model implementation with GPU support and memory optimization."""

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import gc


def sigm(x):
    """Sigmoid-like function for BIGCLAM."""
    return torch.div(torch.exp(-1. * x), 1. - torch.exp(-1. * x))


def log_likelihood(F, A):
    """
    Compute log-likelihood for BIGCLAM model.
    
    Args:
        F (torch.Tensor): Membership strength matrix (N x C).
        A (torch.Tensor): Adjacency matrix (N x N).
        
    Returns:
        torch.Tensor: Log-likelihood value.
    """
    A_soft = F @ F.T  # Matrix multiplication in PyTorch
    # Clip A_soft to prevent overflow in exp
    A_soft_clipped = torch.clamp(A_soft, max=50)
    FIRST_PART = A * torch.log(1. - torch.exp(-1. * A_soft_clipped) + 1e-10)
    sum_edges = torch.sum(FIRST_PART)
    SECOND_PART = (1 - A) * A_soft
    sum_nedges = torch.sum(SECOND_PART)
    log_likeli = sum_edges - sum_nedges
    return log_likeli


class BIGCLAM:
    """BIGCLAM model class."""
    
    def __init__(self, device=None):
        """
        Initialize BIGCLAM model.
        
        Args:
            device (torch.device, optional): Device to run computation on. 
                                            If None, uses GPU if available.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
    
    def train(self, A, max_communities=10, iterations=100, lr=0.08):
        """
        Train BIGCLAM model to find optimal number of communities.
        Memory-efficient version supporting sparse matrices.
        
        Args:
            A (np.ndarray, scipy.sparse matrix, or torch.Tensor): Adjacency matrix (N x N).
            max_communities (int): Maximum number of communities to try.
            iterations (int): Number of optimization iterations per community number.
            lr (float): Learning rate for optimizer.
            
        Returns:
            tuple: (best_F, best_num_communities) where best_F is the membership 
                   strength matrix and best_num_communities is the optimal number of communities.
        """
        # Convert sparse matrix to dense if needed (memory efficient for PyTorch)
        if issparse(A):
            print("Converting sparse adjacency matrix to dense for training...")
            A = A.toarray()
            gc.collect()
        
        # Convert to tensor if needed
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float32, device=self.device)
        else:
            A = A.to(self.device)
        
        N = A.shape[0]
        best_F = None
        best_num_communities = 1
        best_aic = float('inf')
        
        for num_communities in range(1, max_communities + 1):
            # Initialize membership strength matrix
            F = torch.rand((N, num_communities), device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([F], lr=lr)
            
            for n in range(iterations):
                optimizer.zero_grad()
                ll = log_likelihood(F, A)
                loss = -ll  # Minimize negative log-likelihood
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    # Ensure F is nonnegative
                    F.data = torch.clamp(F.data, min=1e-12)
            
            # Calculate AIC
            k = N * num_communities  # Number of parameters
            aic = -2 * ll.item() + 2 * k
            
            if aic < best_aic:
                best_aic = aic
                best_F = F.detach().cpu().numpy()
                best_num_communities = num_communities
            
            print(f'At step {num_communities}/{max_communities}, AIC: {aic:.4f}')
            
            # Clean up GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"Best number of communities: {best_num_communities} (AIC: {best_aic:.4f})")
        return best_F, best_num_communities
    
    def visualize_bipartite(self, F, p2c=None, save_path=None):
        """
        Visualize bipartite graph of people and communities.
        
        Args:
            F (np.ndarray): Membership strength matrix (N x C).
            p2c (list, optional): Community assignments for visualization.
            save_path (str, optional): Path to save the figure.
        """
        B = nx.Graph()
        N, C = F.shape
        
        person_nodes = [f"Person {i}" for i in range(N)]
        community_nodes = [f"Community {i}" for i in range(C)]
        
        B.add_nodes_from(person_nodes, bipartite=0)
        B.add_nodes_from(community_nodes, bipartite=1)
        
        for i, comm_prefs in enumerate(F):
            for c, membership in enumerate(comm_prefs):
                if membership > 0.01:  # Threshold for visualization
                    B.add_edge(person_nodes[i], community_nodes[c])
        
        pos = nx.drawing.layout.bipartite_layout(B, person_nodes)
        plt.figure(figsize=(12, 8))
        nx.draw(
            B, pos, with_labels=True,
            node_color=['skyblue' if n in person_nodes else 'lightgreen' for n in B.nodes()],
            edge_color='gray'
        )
        plt.title('Bipartite Graph of People and Communities')
        
        if save_path:
            plt.savefig(save_path, format='svg', bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def train_bigclam(A, max_communities=10, iterations=100, lr=0.08, device=None):
    """
    Convenience function to train BIGCLAM model.
    
    Args:
        A (np.ndarray or torch.Tensor): Adjacency matrix.
        max_communities (int): Maximum number of communities to try.
        iterations (int): Number of optimization iterations.
        lr (float): Learning rate.
        device (torch.device, optional): Device to use.
        
    Returns:
        tuple: (best_F, best_num_communities) membership matrix and optimal communities.
    """
    model = BIGCLAM(device=device)
    return model.train(A, max_communities, iterations, lr)

