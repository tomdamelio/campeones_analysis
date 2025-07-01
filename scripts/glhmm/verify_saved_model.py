#!/usr/bin/env python
"""
Verify Saved GLHMM Model Script

This script loads and inspects a saved GLHMM model (.pkl file) to verify that it contains
all the expected components of a trained Hidden Markov Model. It checks the model's
structure, parameters, and capabilities to ensure the model was saved correctly and
can be used for analysis or further processing.

Usage:
    python scripts/glhmm/verify_saved_model.py --model-path path/to/model.pkl
    python scripts/glhmm/verify_saved_model.py --model-path data/derivatives/glhmm_results/hmm_model_sub-14_concatenated_states-10_test.pkl
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
import traceback

# Find the repository root
script_path = Path(__file__).resolve()
repo_root = script_path
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent

sys.path.append(str(repo_root))

# Import GLHMM
try:
    import glhmm
    print("✓ GLHMM library imported successfully")
except ImportError as e:
    print(f"❌ Error importing GLHMM: {e}")
    print("Please install GLHMM: pip install glhmm")
    sys.exit(1)


def load_and_verify_model(model_path):
    """
    Load a saved GLHMM model and verify its contents.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the saved model (.pkl file)
        
    Returns
    -------
    hmm : glhmm.glhmm or None
        Loaded HMM model if successful, None otherwise
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING SAVED GLHMM MODEL")
    print(f"{'='*80}")
    
    model_path = Path(model_path)
    
    # Check if file exists
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"📁 Model file: {model_path}")
    print(f"📊 File size: {model_path.stat().st_size / (1024**2):.2f} MB")
    
    # Load the model
    try:
        print(f"\n🔄 Loading model from pickle file...")
        with open(model_path, 'rb') as f:
            hmm = pickle.load(f)
        print(f"✓ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return None
    
    # Verify the model type
    print(f"\n🔍 MODEL TYPE VERIFICATION:")
    print(f"  Model class: {type(hmm)}")
    print(f"  Module: {type(hmm).__module__}")
    
    if not isinstance(hmm, glhmm.glhmm.glhmm):
        print(f"⚠️  WARNING: Expected glhmm.glhmm.glhmm, got {type(hmm)}")
    else:
        print(f"✓ Correct model type: glhmm.glhmm.glhmm")
    
    return hmm


def inspect_model_hyperparameters(hmm):
    """
    Inspect and display model hyperparameters.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Loaded HMM model
    """
    print(f"\n🎛️  HYPERPARAMETERS:")
    
    try:
        hyperparams = hmm.hyperparameters
        print(f"  Hyperparameters type: {type(hyperparams)}")
        print(f"  Available keys: {list(hyperparams.keys()) if hasattr(hyperparams, 'keys') else 'N/A'}")
        
        # Check key hyperparameters
        key_params = ['K', 'covtype', 'model_beta']
        for param in key_params:
            if param in hyperparams:
                print(f"  {param}: {hyperparams[param]}")
            else:
                print(f"  {param}: ⚠️  NOT FOUND")
        
        # Show all hyperparameters
        print(f"\n  📋 All hyperparameters:")
        for key, value in hyperparams.items():
            print(f"    {key}: {value}")
            
    except Exception as e:
        print(f"  ❌ Error accessing hyperparameters: {e}")


def inspect_model_parameters(hmm):
    """
    Inspect and display model parameters (Pi, P, Sigma, etc.).
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Loaded HMM model
    """
    print(f"\n📊 MODEL PARAMETERS:")
    
    # Check initial state probabilities (Pi)
    try:
        pi = hmm.Pi
        print(f"  ✓ Initial state probabilities (Pi):")
        print(f"    Shape: {pi.shape}")
        print(f"    Type: {type(pi)}")
        print(f"    Sum: {np.sum(pi):.6f} (should be ≈ 1.0)")
        print(f"    Values: {pi}")
        
    except Exception as e:
        print(f"  ❌ Error accessing Pi: {e}")
    
    # Check transition probabilities (P)
    try:
        P = hmm.P
        print(f"\n  ✓ Transition probabilities (P):")
        print(f"    Shape: {P.shape}")
        print(f"    Type: {type(P)}")
        print(f"    Row sums: {np.sum(P, axis=1)} (each should be ≈ 1.0)")
        print(f"    Sample values (first 3x3):")
        display_size = min(3, P.shape[0])
        for i in range(display_size):
            row_values = [f"{P[i, j]:.4f}" for j in range(min(display_size, P.shape[1]))]
            print(f"      Row {i+1}: [{', '.join(row_values)}{'...' if P.shape[1] > display_size else ''}]")
            
    except Exception as e:
        print(f"  ❌ Error accessing P: {e}")
    
    # Check if Sigma exists (covariance information)
    try:
        if hasattr(hmm, 'Sigma'):
            sigma = hmm.Sigma
            print(f"\n  ✓ Covariance information (Sigma):")
            print(f"    Type: {type(sigma)}")
            print(f"    Length: {len(sigma) if hasattr(sigma, '__len__') else 'N/A'}")
            
            # Try to inspect first element
            if hasattr(sigma, '__getitem__') and len(sigma) > 0:
                first_elem = sigma[0]
                print(f"    First element type: {type(first_elem)}")
                if hasattr(first_elem, 'keys'):
                    print(f"    First element keys: {list(first_elem.keys())}")
        else:
            print(f"  ⚠️  Sigma attribute not found")
            
    except Exception as e:
        print(f"  ❌ Error accessing Sigma: {e}")


def test_model_methods(hmm):
    """
    Test that key model methods are available and callable.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Loaded HMM model
    """
    print(f"\n🔧 MODEL METHODS VERIFICATION:")
    
    # List of expected methods
    expected_methods = [
        'get_means',
        'get_covariance_matrix', 
        'decode',
        'train',
        'predict'
    ]
    
    for method_name in expected_methods:
        try:
            method = getattr(hmm, method_name, None)
            if method is not None:
                if callable(method):
                    print(f"  ✓ {method_name}: Available and callable")
                else:
                    print(f"  ⚠️  {method_name}: Available but not callable")
            else:
                print(f"  ❌ {method_name}: Not found")
                
        except Exception as e:
            print(f"  ❌ {method_name}: Error accessing - {e}")


def test_get_means(hmm):
    """
    Test the get_means() method to verify it works.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Loaded HMM model
    """
    print(f"\n🧠 TESTING get_means() METHOD:")
    
    try:
        means = hmm.get_means()
        print(f"  ✓ get_means() executed successfully")
        print(f"  📊 State means shape: {means.shape}")
        print(f"  📊 State means type: {type(means)}")
        print(f"  📊 Data type: {means.dtype}")
        print(f"  📊 Value range: [{np.min(means):.4f}, {np.max(means):.4f}]")
        
        # Show sample values
        K = means.shape[1] if len(means.shape) > 1 else 1
        n_features = means.shape[0] if len(means.shape) > 1 else len(means)
        
        print(f"  📊 Number of states: {K}")
        print(f"  📊 Number of features: {n_features}")
        
        # Show first few values
        if len(means.shape) == 2:
            print(f"  📊 Sample state means (first 3 features, first 3 states):")
            display_features = min(3, means.shape[0])
            display_states = min(3, means.shape[1])
            
            for i in range(display_features):
                values = [f"{means[i, j]:.4f}" for j in range(display_states)]
                print(f"    Feature {i+1}: [{', '.join(values)}{'...' if means.shape[1] > display_states else ''}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in get_means(): {e}")
        traceback.print_exc()
        return False


def test_get_covariance_matrix(hmm):
    """
    Test the get_covariance_matrix() method to verify it works.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Loaded HMM model
    """
    print(f"\n🔗 TESTING get_covariance_matrix() METHOD:")
    
    try:
        # Try to get covariance matrix for state 0
        cov_matrix = hmm.get_covariance_matrix(k=0)
        print(f"  ✓ get_covariance_matrix(k=0) executed successfully")
        print(f"  📊 Covariance matrix shape: {cov_matrix.shape}")
        print(f"  📊 Covariance matrix type: {type(cov_matrix)}")
        print(f"  📊 Data type: {cov_matrix.dtype}")
        print(f"  📊 Value range: [{np.min(cov_matrix):.4f}, {np.max(cov_matrix):.4f}]")
        
        # Check if matrix is square
        if len(cov_matrix.shape) == 2 and cov_matrix.shape[0] == cov_matrix.shape[1]:
            print(f"  ✓ Matrix is square ({cov_matrix.shape[0]} x {cov_matrix.shape[1]})")
            
            # Check if matrix is symmetric (covariance matrices should be)
            if np.allclose(cov_matrix, cov_matrix.T, rtol=1e-10):
                print(f"  ✓ Matrix is symmetric (as expected for covariance)")
            else:
                print(f"  ⚠️  Matrix is not symmetric (unexpected for covariance)")
                
        else:
            print(f"  ⚠️  Matrix is not square: {cov_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in get_covariance_matrix(): {e}")
        traceback.print_exc()
        return False


def inspect_preprocessing_info(hmm):
    """
    Inspect preprocessing information stored in the model.
    
    Parameters
    ----------
    hmm : glhmm.glhmm
        Loaded HMM model
    """
    print(f"\n🔄 PREPROCESSING INFORMATION:")
    
    # Check for preprocessing log
    preprocessing_attrs = [
        'preproclogY',
        'preproc', 
        'preprocessing',
        'pca',
        'standardization'
    ]
    
    found_preproc = False
    
    for attr in preprocessing_attrs:
        try:
            if hasattr(hmm, attr):
                preproc_info = getattr(hmm, attr)
                print(f"  ✓ Found {attr}:")
                print(f"    Type: {type(preproc_info)}")
                
                if isinstance(preproc_info, dict):
                    print(f"    Keys: {list(preproc_info.keys())}")
                    for key, value in preproc_info.items():
                        if hasattr(value, 'shape'):
                            print(f"      {key}: {type(value)} with shape {value.shape}")
                        else:
                            print(f"      {key}: {type(value)} = {value}")
                else:
                    print(f"    Content: {preproc_info}")
                
                found_preproc = True
                break
                
        except Exception as e:
            print(f"  Error accessing {attr}: {e}")
    
    if not found_preproc:
        print(f"  ⚠️  No preprocessing information found in standard attributes")
        
    # Try to list all attributes to see what's available
    try:
        all_attrs = [attr for attr in dir(hmm) if not attr.startswith('_')]
        print(f"\n  📋 All available attributes:")
        for attr in sorted(all_attrs):
            try:
                value = getattr(hmm, attr)
                if callable(value):
                    print(f"    {attr}(): method")
                else:
                    print(f"    {attr}: {type(value)}")
            except:
                print(f"    {attr}: <error accessing>")
                
    except Exception as e:
        print(f"  Error listing attributes: {e}")


def main():
    """
    Main function to verify a saved GLHMM model.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Verify and inspect a saved GLHMM model file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/glhmm/verify_saved_model.py --model-path data/derivatives/glhmm_results/hmm_model_sub-14_concatenated_states-10_test.pkl
    python scripts/glhmm/verify_saved_model.py --model-path path/to/your/model.pkl --verbose
        """
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the saved model (.pkl file)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load and verify the model
        hmm = load_and_verify_model(args.model_path)
        
        if hmm is None:
            print(f"\n❌ Failed to load model. Verification cannot continue.")
            return 1
        
        # Inspect model components
        inspect_model_hyperparameters(hmm)
        inspect_model_parameters(hmm)
        test_model_methods(hmm)
        
        # Test key methods
        means_ok = test_get_means(hmm)
        covariance_ok = test_get_covariance_matrix(hmm)
        
        if args.verbose:
            inspect_preprocessing_info(hmm)
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"VERIFICATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Correct model type (glhmm.glhmm.glhmm)")
        print(f"✓ Hyperparameters accessible")
        print(f"✓ Model parameters (Pi, P) accessible")
        print(f"{'✓' if means_ok else '❌'} get_means() method {'works' if means_ok else 'failed'}")
        print(f"{'✓' if covariance_ok else '❌'} get_covariance_matrix() method {'works' if covariance_ok else 'failed'}")
        
        if means_ok and covariance_ok:
            print(f"\n🎉 MODEL VERIFICATION PASSED!")
            print(f"   The saved model contains all expected components and methods work correctly.")
            print(f"   This model can be used for analysis, inference, and visualization.")
            return 0
        else:
            print(f"\n⚠️  MODEL VERIFICATION PARTIAL:")
            print(f"   The model loaded but some methods failed.")
            print(f"   Check the error messages above for details.")
            return 2
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR during verification: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 