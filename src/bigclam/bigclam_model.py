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
    
    def train(self, A, max_communities=10, min_communities=1, iterations=100, lr=0.08, criterion='BIC',
              adaptive_lr=True, adaptive_iterations=True, early_stopping=True,
              convergence_threshold=1e-6, patience=10, num_restarts=1):
        """
        Train BIGCLAM model to automatically find optimal number of communities using AIC/BIC.
        
        The model tests communities from min_communities to max_communities and automatically selects
        the optimal number based on Akaike Information Criterion (AIC) or Bayesian 
        Information Criterion (BIC). Lower score indicates better model fit with penalty 
        for complexity.
        
        - BIC: Uses log(N) × k penalty - stronger penalty, better for smaller datasets
        - AIC: Uses 2 × k penalty - less penalty, better for larger datasets
        
        Memory-efficient version supporting sparse matrices.
        
        Args:
            A (np.ndarray, scipy.sparse matrix, or torch.Tensor): Adjacency matrix (N x N).
            max_communities (int): Maximum number of communities to test.
            min_communities (int): Minimum number of communities to test (default: 1).
            iterations (int): Base number of optimization iterations per community number.
            lr (float): Base learning rate for Adam optimizer.
            criterion (str): Model selection criterion - 'AIC' or 'BIC' (default: 'BIC')
            adaptive_lr (bool): Automatically adjust learning rate based on graph size (default: True)
            adaptive_iterations (bool): Automatically adjust iterations based on graph size and community number (default: True)
            early_stopping (bool): Enable early stopping when convergence detected (default: True)
            convergence_threshold (float): Loss change threshold for convergence (default: 1e-6)
            patience (int): Number of iterations without improvement before stopping (default: 10)
            num_restarts (int): Number of random restarts per community number (default: 1)
            
        Returns:
            tuple: (best_F, best_num_communities) where:
                - best_F: Membership strength matrix (N x optimal_C) for optimal number of communities
                - best_num_communities: Automatically determined optimal number of communities (≥ min_communities)
        """
        criterion = criterion.upper()  # Normalize to uppercase
        if criterion not in ['AIC', 'BIC']:
            print(f"[WARNING] Invalid criterion '{criterion}', using 'BIC'")
            criterion = 'BIC'
        
        # Convert parameters to correct types
        convergence_threshold = float(convergence_threshold)
        patience = int(patience)
        num_restarts = int(num_restarts)
        
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
        
        # Adaptive learning rate based on graph size
        if adaptive_lr:
            if N < 1000:
                adjusted_lr = lr
            elif N < 2000:
                adjusted_lr = lr * 0.75
            else:
                adjusted_lr = lr * 0.5
            if adjusted_lr != lr:
                print(f"[Adaptive LR] Graph size {N}: adjusted LR from {lr} to {adjusted_lr}")
        else:
            adjusted_lr = lr
        
        # Ensure min_communities <= max_communities
        if min_communities > max_communities:
            print(f"[WARNING] min_communities ({min_communities}) > max_communities ({max_communities}), setting min_communities = max_communities")
            min_communities = max_communities
        
        best_F = None
        best_num_communities = min_communities
        best_score = float('inf')
        
        criterion_name = "Akaike" if criterion == 'AIC' else "Bayesian"
        penalty_formula = "2 × k" if criterion == 'AIC' else "log(N) × k"
        print(f"\n[Searching optimal communities] Testing {min_communities} to {max_communities} communities...")
        if min_communities > 1:
            print(f"[Note] Minimum communities set to {min_communities} for finer-grained subtyping")
        print(f"Using {criterion} ({criterion_name} Information Criterion) for model selection")
        print(f"(Lower {criterion} = better fit with complexity penalty: {penalty_formula})")
        if criterion == 'BIC':
            print(f"BIC uses stronger penalty than AIC, better for smaller datasets")
        else:
            print(f"AIC uses less penalty than BIC, better for larger datasets")
        
        if adaptive_iterations:
            print(f"Adaptive iterations: enabled (more iterations for larger graphs/communities)")
        if early_stopping:
            print(f"Early stopping: enabled (patience={patience}, threshold={convergence_threshold})")
        if num_restarts > 1:
            print(f"Multiple restarts: {num_restarts} per community number")
        print()
        
        for num_communities in range(min_communities, max_communities + 1):
            # Adaptive iterations based on graph size and community number
            if adaptive_iterations:
                if N < 1000:
                    adjusted_iterations = iterations
                elif N < 2000:
                    adjusted_iterations = int(iterations * 1.5)
                else:
                    adjusted_iterations = int(iterations * 2.0)
                # Increase iterations for higher community numbers (10% more per community)
                adjusted_iterations = int(adjusted_iterations * (1 + 0.1 * (num_communities - 1)))
            else:
                adjusted_iterations = iterations
            
            best_restart_F = None
            best_restart_score = float('inf')
            
            # Multiple random restarts
            for restart in range(num_restarts):
                if num_restarts > 1:
                    print(f"  Restart {restart + 1}/{num_restarts} for {num_communities} communities...")
                
            # Initialize membership strength matrix
            F = torch.rand((N, num_communities), device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([F], lr=adjusted_lr)
                
            best_loss = float('inf')
            no_improvement_count = 0
            
            for n in range(adjusted_iterations):
                optimizer.zero_grad()
                ll = log_likelihood(F, A)
                loss = -ll  # Minimize negative log-likelihood
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    # Ensure F is nonnegative
                    F.data = torch.clamp(F.data, min=1e-12)
            
                    # Early stopping check
                    if early_stopping:
                        loss_value = loss.item()
                        if loss_value < best_loss - convergence_threshold:
                            best_loss = loss_value
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                            if no_improvement_count >= patience:
                                if num_restarts == 1:  # Only print if not doing multiple restarts
                                    print(f"    Early stopping at iteration {n+1}/{adjusted_iterations}")
                                break
                
                # Calculate score based on criterion
            k = N * num_communities  # Number of parameters
            if criterion == 'BIC':
                    # BIC: BIC = -2 * log_likelihood + log(N) * k
                score = -2 * ll.item() + np.log(N) * k
            else:
                    # AIC: AIC = -2 * log_likelihood + 2 * k
                score = -2 * ll.item() + 2 * k
                
                # Track best restart for this community number
            if score < best_restart_score:
                best_restart_score = score
                best_restart_F = F.detach().cpu().numpy()
                
                # Clean up GPU memory after each restart
            if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Use best restart result
            F_final = best_restart_F
            score_final = best_restart_score
            
            is_best = score_final < best_score
            if is_best:
                best_score = score_final
                best_F = F_final
                best_num_communities = num_communities
            
            status = "★ BEST" if is_best else ""
            print(f'  Communities={num_communities:2d}/{max_communities}, {criterion}: {score_final:.4f} {status}')
            
            # Clean up GPU memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        print(f"\n[Optimal Result] Best number of communities: {best_num_communities} ({criterion}: {best_score:.4f})")
        print(f"This value was automatically selected based on {criterion} model selection.")
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


def train_bigclam(A, max_communities=10, min_communities=1, iterations=100, lr=0.08, device=None, criterion='BIC',
                  adaptive_lr=True, adaptive_iterations=True, early_stopping=True,
                  convergence_threshold=1e-6, patience=10, num_restarts=1):
    """
    Convenience function to train BIGCLAM model with automatic community selection.
    
    Automatically finds optimal number of communities using AIC/BIC by testing
    all values from min_communities to max_communities and selecting the best.
    
    Args:
        A (np.ndarray or torch.Tensor): Adjacency matrix (N x N).
        max_communities (int): Maximum number of communities to test.
        min_communities (int): Minimum number of communities to test (default: 1).
                              Set to 5 or higher for finer-grained subtyping than PAM50.
        iterations (int): Base number of optimization iterations per community number.
        lr (float): Base learning rate for Adam optimizer.
        device (torch.device, optional): Device to use (GPU if available, else CPU).
        criterion (str): Model selection criterion - 'AIC' or 'BIC' (default: 'BIC')
        adaptive_lr (bool): Automatically adjust learning rate based on graph size (default: True)
        adaptive_iterations (bool): Automatically adjust iterations based on graph size (default: True)
        early_stopping (bool): Enable early stopping when convergence detected (default: True)
        convergence_threshold (float): Loss change threshold for convergence (default: 1e-6)
        patience (int): Number of iterations without improvement before stopping (default: 10)
        num_restarts (int): Number of random restarts per community number (default: 1)
        
    Returns:
        tuple: (best_F, best_num_communities) where:
            - best_F: Membership strength matrix for optimal number of communities
            - best_num_communities: Automatically determined optimal number (via AIC/BIC, ≥ min_communities)
    """
    model = BIGCLAM(device=device)
    return model.train(A, max_communities, min_communities, iterations, lr, criterion,
                      adaptive_lr, adaptive_iterations, early_stopping,
                      convergence_threshold, patience, num_restarts)

