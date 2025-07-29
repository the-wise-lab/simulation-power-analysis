#!/usr/bin/env python3
"""
Quick test script for simulation-power-analysis package.

This is a minimal test that quickly verifies the basic functionality
of the power analysis package with small sample sizes and iterations.

Usage:
    python quick_test.py
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: Import MPI for rank logic
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    MPI_AVAILABLE = False
    comm = None
    rank = 0

# Import the analysis functions
from simulation_power_analysis.analysis import (
    fit_distribution, 
    sample_synthetic, 
    power_analysis
)

# Set random seed for reproducibility
np.random.seed(42)

def create_simple_data(n=200):
    """Create a simple dataset for quick testing."""
    
    # Simple data with clear relationships
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.binomial(1, 0.5, n)  # Binary variable
    x3 = np.random.normal(5, 2, n)
    
    # Outcome with known relationships
    y = 2 + 1.5*x1 + 2*x2 + 0.5*x3 + np.random.normal(0, 1, n)
    
    data = pd.DataFrame({
        'x1': x1,
        'x2': x2, 
        'x3': x3,
        'y': y
    })
    
    return data


def quick_test():
    """Run a quick test of the main functionality."""

    if rank == 0:
        print("Running quick test of simulation-power-analysis...")

    # Create test data (all ranks, or only rank 0 and broadcast)
    if rank == 0:
        print("1. Creating test data...")
        data = create_simple_data(200)
        print(f"   Data shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
    else:
        data = None

    # Broadcast data to all ranks if using MPI
    if MPI_AVAILABLE:
        data = comm.bcast(data, root=0)

    # Test distribution fitting (all ranks, but only rank 0 prints)
    if rank == 0:
        print("2. Testing distribution fitting...")
    try:
        dist = fit_distribution(data)
        if rank == 0:
            print("   ✓ Distribution fitting successful")
    except Exception as e:
        if rank == 0:
            print(f"   ✗ Distribution fitting failed: {e}")
        return False

    # Test synthetic data generation (all ranks, only rank 0 prints)
    if rank == 0:
        print("3. Testing synthetic data generation...")
    try:
        synthetic = sample_synthetic(dist, 100, binary_columns=['x2'])
        if rank == 0:
            print(f"   ✓ Synthetic data generated, shape: {synthetic.shape}")
    except Exception as e:
        if rank == 0:
            print(f"   ✗ Synthetic data generation failed: {e}")
        return False

    # Test power analysis (small scale)
    if rank == 0:
        print("4. Testing power analysis...")
    try:
        sample_sizes = np.array([250, 500, 750])
        # For quick test, use_mpi can be set to True if you want to test MPI
        results = power_analysis(
            data=data,
            dependent_var='y',
            variables=['x1', 'x2', 'x3'],
            sample_sizes=sample_sizes,
            n_iter=200,  # Very small for quick test
            binary_cols=['x2'],
            n_jobs=1,
            use_mpi=MPI_AVAILABLE
        )
        if rank == 0:
            print(f"   ✓ Power analysis completed, results shape: {results.shape}")
            print("   Sample results:")
            print(results.head())
    except Exception as e:
        if rank == 0:
            print(f"   ✗ Power analysis failed: {e}")
        return False

    if rank == 0:
        print("\n✓ All tests passed! The package is working correctly.")
    return True

if __name__ == "__main__":
    success = quick_test()
    if rank == 0:
        if success:
            print("\nQuick test completed successfully!")
        else:
            print("\nQuick test failed. Check the error messages above.")
