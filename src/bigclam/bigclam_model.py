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
        Train BIGCLAM model to automatically find optimal number of communities using BIC.
        
        The model tests communities from 1 to max_communities and automatically selects
        the optimal number based on Bayesian Information Criterion (BIC). Lower BIC indicates
        better model fit with penalty for complexity. BIC uses a stronger penalty than AIC,
        making it more suitable for community detection as it better balances fit and complexity.
        
        Memory-efficient version supporting sparse matrices.
        
        Args:
            A (np.ndarray, scipy.sparse matrix, or torch.Tensor): Adjacency matrix (N x N).
            max_communities (int): Maximum number of communities to test (searches 1 to this value).
            iterations (int): Number of optimization iterations per community number.
            lr (float): Learning rate for Adam optimizer.
            
        Returns:
            tuple: (best_F, best_num_communities) where:
                - best_F: Membership strength matrix (N x optimal_C) for optimal number of communities
                - best_num_communities: Automatically determined optimal number of communities (via BIC)
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
        best_bic = float('inf')
        
        print(f"\n[Searching optimal communities] Testing 1 to {max_communities} communities...")
        print(f"Using BIC (Bayesian Information Criterion) for model selection")
        print(f"(Lower BIC = better fit with complexity penalty: log(N) × parameters)")
        print(f"BIC uses stronger penalty than AIC, better for community detection\n")
        
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
            
            # Calculate BIC: BIC = -2 * log_likelihood + log(N) * k
            # BIC uses log(N) penalty instead of constant 2, providing stronger penalty for larger N
            k = N * num_communities  # Number of parameters
            bic = -2 * ll.item() + np.log(N) * k
            
            is_best = bic < best_bic
            if is_best:
                best_bic = bic
                best_F = F.detach().cpu().numpy()
                best_num_communities = num_communities
            
            status = "★ BEST" if is_best else ""
            print(f'  Communities={num_communities:2d}/{max_communities}, BIC: {bic:.4f} {status}')
            
            # Clean up GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\n[Optimal Result] Best number of communities: {best_num_communities} (BIC: {best_bic:.4f})")
        print(f"This value was automatically selected based on BIC model selection.")
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
    Convenience function to train BIGCLAM model with automatic community selection.
    
    Automatically finds optimal number of communities using BIC by testing
    all values from 1 to max_communities and selecting the best.
    
    Args:
        A (np.ndarray or torch.Tensor): Adjacency matrix (N x N).
        max_communities (int): Maximum number of communities to test (searches 1 to this value).
        iterations (int): Number of optimization iterations per community number.
        lr (float): Learning rate for Adam optimizer.
        device (torch.device, optional): Device to use (GPU if available, else CPU).
        
    Returns:
        tuple: (best_F, best_num_communities) where:
            - best_F: Membership strength matrix for optimal number of communities
            - best_num_communities: Automatically determined optimal number (via BIC)
    """
    model = BIGCLAM(device=device)
    return model.train(A, max_communities, iterations, lr)

