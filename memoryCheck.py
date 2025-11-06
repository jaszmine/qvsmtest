import numpy as np
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2, ceil

def calculate_quantum_memory_requirements(qubits):
    """
    Calculate memory requirements for quantum state simulation
    """
    # Memory for state vector (complex128 = 16 bytes per amplitude)
    state_vector_memory = (2 ** qubits) * 16  # bytes
    
    # Memory for unitary matrix (for some operations)
    unitary_memory = (2 ** (2 * qubits)) * 16  # bytes
    
    # Memory for density matrix (if needed)
    density_matrix_memory = (2 ** (2 * qubits)) * 16  # bytes
    
    return {
        'qubits': qubits,
        'state_vector_bytes': state_vector_memory,
        'state_vector_GB': state_vector_memory / (1024**3),
        'unitary_matrix_bytes': unitary_memory,
        'unitary_matrix_GB': unitary_memory / (1024**3),
        'density_matrix_bytes': density_matrix_memory,
        'density_matrix_GB': density_matrix_memory / (1024**3)
    }

def get_system_memory_info():
    """Get available system memory"""
    virtual_memory = psutil.virtual_memory()
    return {
        'total_GB': virtual_memory.total / (1024**3),
        'available_GB': virtual_memory.available / (1024**3),
        'used_GB': virtual_memory.used / (1024**3),
        'free_GB': virtual_memory.free / (1024**3)
    }

def find_max_qubits(max_qubits=50, memory_limit_GB=None):
    """
    Find maximum number of qubits that can be simulated with available memory
    """
    if memory_limit_GB is None:
        memory_info = get_system_memory_info()
        # Use 80% of available memory as safe limit
        memory_limit_GB = memory_info['available_GB'] * 0.8
    
    print("=" * 70)
    print("QUANTUM MEMORY REQUIREMENTS ANALYSIS")
    print("=" * 70)
    
    # Display system memory information
    memory_info = get_system_memory_info()
    print(f"\nSYSTEM MEMORY INFORMATION:")
    print(f"  Total RAM:      {memory_info['total_GB']:.1f} GB")
    print(f"  Available RAM:  {memory_info['available_GB']:.1f} GB")
    print(f"  Used RAM:       {memory_info['used_GB']:.1f} GB")
    print(f"  Free RAM:       {memory_info['free_GB']:.1f} GB")
    print(f"  Safe limit:     {memory_limit_GB:.1f} GB (80% of available)")
    
    # Calculate memory requirements for different qubit counts
    results = []
    max_feasible_qubits = 0
    
    print(f"\n{'Qubits':>6} {'State Vector':>15} {'Unitary Matrix':>15} {'Density Matrix':>15} {'Status':>10}")
    print("-" * 70)
    
    for n_qubits in range(1, max_qubits + 1):
        req = calculate_quantum_memory_requirements(n_qubits)
        
        # Check if state vector fits in memory (most common requirement)
        state_vector_fits = req['state_vector_GB'] <= memory_limit_GB
        unitary_fits = req['unitary_matrix_GB'] <= memory_limit_GB
        density_fits = req['density_matrix_GB'] <= memory_limit_GB
        
        status = []
        if state_vector_fits:
            status.append("State✓")
            max_feasible_qubits = n_qubits
        if unitary_fits:
            status.append("Unitary✓")
        if density_fits:
            status.append("Density✓")
        
        status_str = ", ".join(status) if status else "TOO LARGE"
        
        print(f"{n_qubits:6d} {req['state_vector_GB']:12.3f} GB {req['unitary_matrix_GB']:12.3f} GB "
              f"{req['density_matrix_GB']:12.3f} GB {status_str:>10}")
        
        results.append(req)
    
    print("-" * 70)
    print(f"\nMAXIMUM FEASIBLE QUBITS: {max_feasible_qubits}")
    print(f"  - State vector: {calculate_quantum_memory_requirements(max_feasible_qubits)['state_vector_GB']:.3f} GB")
    print(f"  - Safe limit:   {memory_limit_GB:.1f} GB")
    
    return results, max_feasible_qubits

def create_detailed_scaling_table(results, max_feasible_qubits):
    """Create a detailed scaling table"""
    df = pd.DataFrame(results)
    
    # Add some useful derived columns
    df['amplitudes'] = 2 ** df['qubits']
    df['log2_amplitudes'] = df['qubits']
    df['state_vector_MB'] = df['state_vector_bytes'] / (1024**2)
    df['unitary_matrix_MB'] = df['unitary_matrix_bytes'] / (1024**2)
    
    print(f"\n{'='*80}")
    print("DETAILED QUANTUM MEMORY SCALING TABLE")
    print(f"{'='*80}")
    
    # Display key columns in a nice format
    display_cols = ['qubits', 'amplitudes', 'state_vector_GB', 'unitary_matrix_GB', 'density_matrix_GB']
    display_df = df[display_cols].copy()
    display_df.columns = ['Qubits', 'Amplitudes', 'State Vector (GB)', 'Unitary Matrix (GB)', 'Density Matrix (GB)']
    
    # Format the display
    pd.set_option('display.float_format', lambda x: f'{x:.3f}' if x < 1 else f'{x:.1f}')
    pd.set_option('display.max_rows', None)
    
    print(display_df.to_string(index=False))
    
    return df

def plot_memory_scaling(results, max_feasible_qubits):
    """Create visualization of memory scaling"""
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Linear scale
    ax1.plot(df['qubits'], df['state_vector_GB'], 'bo-', linewidth=2, markersize=4, label='State Vector')
    ax1.plot(df['qubits'], df['unitary_matrix_GB'], 'ro-', linewidth=2, markersize=4, label='Unitary Matrix')
    ax1.plot(df['qubits'], df['density_matrix_GB'], 'go-', linewidth=2, markersize=4, label='Density Matrix')
    
    # Add memory limit line
    memory_info = get_system_memory_info()
    safe_limit = memory_info['available_GB'] * 0.8
    ax1.axhline(y=safe_limit, color='red', linestyle='--', alpha=0.7, 
                label=f'Safe Memory Limit ({safe_limit:.1f} GB)')
    
    # Highlight max feasible qubits
    ax1.axvline(x=max_feasible_qubits, color='orange', linestyle=':', alpha=0.8,
                label=f'Max Feasible ({max_feasible_qubits} qubits)')
    
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Memory Required (GB)')
    ax1.set_title('Quantum Memory Requirements - Linear Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax2.semilogy(df['qubits'], df['state_vector_GB'], 'bo-', linewidth=2, markersize=4, label='State Vector')
    ax2.semilogy(df['qubits'], df['unitary_matrix_GB'], 'ro-', linewidth=2, markersize=4, label='Unitary Matrix')
    ax2.semilogy(df['qubits'], df['density_matrix_GB'], 'go-', linewidth=2, markersize=4, label='Density Matrix')
    
    ax2.axhline(y=safe_limit, color='red', linestyle='--', alpha=0.7, 
                label=f'Safe Memory Limit ({safe_limit:.1f} GB)')
    ax2.axvline(x=max_feasible_qubits, color='orange', linestyle=':', alpha=0.8,
                label=f'Max Feasible ({max_feasible_qubits} qubits)')
    
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Memory Required (GB) - Log Scale')
    ax2.set_title('Quantum Memory Requirements - Log Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_memory_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def practical_recommendations(max_feasible_qubits, memory_info):
    """Provide practical recommendations based on analysis"""
    print(f"\n{'='*80}")
    print("PRACTICAL RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\nBased on your system with {memory_info['available_GB']:.1f} GB available RAM:")
    print(f"✓ Maximum feasible qubits: {max_feasible_qubits}")
    print(f"✓ Safe working limit: {max_feasible_qubits - 1} qubits (for buffer)")
    
    print(f"\nRECOMMENDED QUBIT RANGES:")
    print(f"  Beginner (1-12 qubits):    Easy simulation, fast execution")
    print(f"  Intermediate (13-{max_feasible_qubits-5} qubits): Good for algorithms, moderate memory")
    print(f"  Advanced ({max_feasible_qubits-4}-{max_feasible_qubits} qubits): Pushing limits, high memory")
    
    print(f"\nYOUR CURRENT STATUS:")
    print(f"  Your gene expression analysis used: 32 qubits (angle encoding)")
    print(f"  Your amplitude encoding used: 5 qubits (very efficient)")
    print(f"  Memory usage for 32-qubit state vector: {calculate_quantum_memory_requirements(32)['state_vector_GB']:.1f} GB")
    
    if max_feasible_qubits >= 32:
        print(f"  ✓ Your system can handle 32-qubit simulations")
    else:
        print(f"  ✗ Your system cannot handle 32-qubit simulations")
    
    print(f"\nOPTIMIZATION TIPS:")
    print("  1. Use amplitude encoding when possible (reduces qubit count)")
    print("  2. Use statevector simulators only when necessary")
    print("  3. Consider using density matrix for noisy simulations < 16 qubits")
    print("  4. Use Qiskit's Aer simulator with memory optimization")
    print("  5. Close other memory-intensive applications during quantum simulations")

def compare_with_real_hardware():
    """Compare with real quantum hardware capabilities"""
    print(f"\n{'='*80}")
    print("COMPARISON WITH REAL QUANTUM HARDWARE")
    print(f"{'='*80}")
    
    quantum_processors = {
        'IBM Quantum': [127, 433],  # IBM's largest processors
        'Google Sycamore': [53],     # Google's quantum processor
        'Rigetti': [80],            # Rigetti's processors
        'IonQ': [32],               # Ion trap systems
        'Your Simulation': []        # Will be filled
    }
    
    memory_info = get_system_memory_info()
    max_qubits_simulation = 0
    for qubits in range(1, 51):
        if calculate_quantum_memory_requirements(qubits)['state_vector_GB'] <= memory_info['available_GB'] * 0.8:
            max_qubits_simulation = qubits
    
    quantum_processors['Your Simulation'] = [max_qubits_simulation]
    
    print(f"\nQuantum Processor Qubit Counts:")
    for processor, qubits_list in quantum_processors.items():
        if qubits_list:
            print(f"  {processor:20} {max(qubits_list):3d} qubits")
    
    print(f"\nKey Insight:")
    print(f"  Your simulation can handle {max_qubits_simulation} qubits")
    print(f"  This exceeds many current quantum processors!")
    print(f"  However, real quantum hardware doesn't have these memory constraints")

# Main execution
if __name__ == "__main__":
    print("Quantum Memory Requirements Calculator")
    print("Calculating maximum feasible qubits for your system...")
    
    # Get system memory and calculate max qubits
    results, max_feasible_qubits = find_max_qubits(max_qubits=50)
    
    # Create detailed table
    df = create_detailed_scaling_table(results, max_feasible_qubits)
    
    # Create visualization
    plot_memory_scaling(results, max_feasible_qubits)
    
    # Get memory info for recommendations
    memory_info = get_system_memory_info()
    
    # Provide practical recommendations
    practical_recommendations(max_feasible_qubits, memory_info)
    
    # Compare with real hardware
    compare_with_real_hardware()
    
    # Save results to CSV
    df.to_csv('quantum_memory_requirements_analysis.csv', index=False)
    print(f"\n✓ Detailed analysis saved to 'quantum_memory_requirements_analysis.csv'")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Your system can safely simulate up to {max_feasible_qubits} qubits")
    print(f"Memory available: {memory_info['available_GB']:.1f} GB")
    print(f"Memory required for {max_feasible_qubits} qubits: {calculate_quantum_memory_requirements(max_feasible_qubits)['state_vector_GB']:.1f} GB")