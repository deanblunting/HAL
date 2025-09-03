#!/usr/bin/env python3
"""
Parallel processing version of QECC code analysis.
Uses HAL's new parallel processing capabilities for faster execution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from hal.parallel import ParallelHAL, create_qecc_graph_from_info


def get_available_code_families(data_folder='data'):
    """Check which QECC code family CSV files are available without loading them."""
    import os
    
    code_files = {
        'BB': f'{data_folder}/bb_codes.csv', 
        'Tile': f'{data_folder}/tile_codes.csv',
        'Radial': f'{data_folder}/radial_codes.csv'
    }
    
    available_families = {}
    
    for family, csv_path in code_files.items():
        if os.path.exists(csv_path):
            # Just count rows without loading all data
            df = pd.read_csv(csv_path, usecols=[0])  # Only read first column to count
            available_families[family] = {
                'path': csv_path,
                'count': len(df)
            }
            print(f"Available: {len(df)} {family} codes")
        else:
            print(f"Warning: {csv_path} not found, skipping {family} codes")
    
    return available_families


def select_code_families_to_process(available_families):
    """Interactive selection of which code families to process."""
    print("\nAvailable code families:")
    family_names = list(available_families.keys())
    
    for i, family in enumerate(family_names, 1):
        print(f"  {i}. {family} ({available_families[family]['count']} codes)")
    
    print(f"  {len(family_names)+1}. All families")
    
    while True:
        try:
            choice = input("\nSelect families to process (comma-separated numbers or 'all'): ").strip().lower()
            
            if choice == 'all' or choice == str(len(family_names)+1):
                return list(available_families.keys())
            
            # Parse comma-separated choices
            choices = [int(x.strip()) for x in choice.split(',')]
            selected_families = []
            
            for choice_num in choices:
                if 1 <= choice_num <= len(family_names):
                    family = family_names[choice_num - 1]
                    selected_families.append(family)
                else:
                    print(f"Invalid choice: {choice_num}")
                    continue
            
            if selected_families:
                total_codes = sum(available_families[family]['count'] for family in selected_families)
                print(f"\nSelected {total_codes} codes from families: {', '.join(selected_families)}")
                return selected_families
            else:
                print("No valid selections made. Please try again.")
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please enter comma-separated numbers or 'all'.")
            continue


def load_selected_code_families(selected_families, available_families):
    """Load QECC codes from selected CSV files only."""
    selected_codes = []
    
    for family in selected_families:
        if family in available_families:
            csv_path = available_families[family]['path']
            print(f"Loading {family} codes from {csv_path}...")
            
            df = pd.read_csv(csv_path)
            codes = []
            
            for _, row in df.iterrows():
                codes.append({
                    'family': family,
                    'n': int(row['n']),
                    'k': int(row['k']),
                    'd': int(row['d']),
                    'logical_efficiency': row['k'] * row['d']**2 / row['n']
                })
            
            selected_codes.extend(codes)
            print(f"Loaded {len(codes)} {family} codes")
    
    print(f"Total codes loaded: {len(selected_codes)}")
    return selected_codes


def process_codes_parallel(codes, n_processes=None, output_file='results/qecc_results_parallel.json'):
    """Process all codes using parallel HAL processing."""
    
    print(f"Starting parallel processing of {len(codes)} codes...")
    
    # Initialize parallel HAL
    parallel_hal = ParallelHAL(n_processes=n_processes, verbose=True)
    
    # Process all codes in parallel
    start_time = time.time()
    results = parallel_hal.process_batch(codes, create_qecc_graph_from_info)
    total_time = time.time() - start_time
    
    # Extract successful results and convert to expected format
    processed_results = []
    for result in results:
        if result['success']:
            processed_result = {
                'family': result['code_info']['family'],
                'n': result['code_info']['n'],
                'k': result['code_info']['k'], 
                'd': result['code_info']['d'],
                'logical_efficiency': result['code_info']['logical_efficiency'],
                'hardware_cost': result['layout'].hardware_cost,
                'metrics': result['layout'].metrics,
                'processing_time': result['processing_time'],
                'graph_stats': result['graph_stats']
            }
            processed_results.append(processed_result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    # Print summary
    print(f"\nParallel Processing Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successfully processed: {len(processed_results)}/{len(codes)} codes")
    print(f"Average time per code: {total_time/len(codes):.3f}s")
    print(f"Results saved to {output_file}")
    
    return processed_results


def create_visualization(results, output_image='results/qecc_cost_vs_efficiency_parallel.png'):
    """Create hardware cost vs logical efficiency plot with enhanced information."""
    
    # Separate by code family
    families = {}
    for result in results:
        family = result['family']
        if family not in families:
            families[family] = {
                'x': [], 'y': [], 'weights': [], 'sizes': [], 'processing_times': []
            }
        
        families[family]['x'].append(result['logical_efficiency'])
        families[family]['y'].append(result['hardware_cost'])
        families[family]['weights'].append(result['metrics'].get('qecc_weight', 0))
        families[family]['sizes'].append(result['n'])
        families[family]['processing_times'].append(result['processing_time'])
    
    # Create enhanced plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {'BB': '#8B4A9C', 'Tile': '#FF6B6B', 'Radial': '#4ECDC4'}
    
    # Main cost vs efficiency plot
    for family, data in families.items():
        ax1.scatter(data['x'], data['y'], 
                   c=colors.get(family, 'gray'), 
                   label=family, 
                   alpha=0.7, 
                   s=60)
    
    ax1.set_xlabel('Logical efficiency kd²/n')
    ax1.set_ylabel('Hardware cost C_hw')
    ax1.set_title('Hardware Cost vs Logical Efficiency\n(Parallel HAL Results)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QECC weight distribution
    for family, data in families.items():
        ax2.scatter(data['x'], data['weights'], 
                   c=colors.get(family, 'gray'), 
                   label=family, 
                   alpha=0.7, 
                   s=60)
    
    ax2.set_xlabel('Logical efficiency kd²/n')
    ax2.set_ylabel('QECC Weight')
    ax2.set_title('QECC Weight vs Logical Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Processing time vs code size
    for family, data in families.items():
        ax3.scatter(data['sizes'], data['processing_times'], 
                   c=colors.get(family, 'gray'), 
                   label=family, 
                   alpha=0.7, 
                   s=60)
    
    ax3.set_xlabel('Number of qubits (n)')
    ax3.set_ylabel('Processing time (s)')
    ax3.set_title('Processing Time vs Code Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Hardware cost histogram
    all_costs = [result['hardware_cost'] for result in results]
    ax4.hist(all_costs, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Hardware cost C_hw')
    ax4.set_ylabel('Number of codes')
    ax4.set_title('Hardware Cost Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Enhanced visualization saved to {output_image}")
    
    # Print detailed statistics
    print(f"\nDetailed Statistics:")
    for family, data in families.items():
        print(f"\n{family}:")
        print(f"  Count: {len(data['x'])}")
        print(f"  Avg QECC weight: {np.mean(data['weights']):.1f}")
        print(f"  Avg hardware cost: {np.mean(data['y']):.2f}")
        print(f"  Avg processing time: {np.mean(data['processing_times']):.3f}s")
        print(f"  Efficiency range: {min(data['x']):.1f} - {max(data['x']):.1f}")
    
    plt.show()
    return families


def main():
    """Main parallel processing pipeline."""
    print("Parallel QECC Code Processing Pipeline")
    print("=" * 45)
    
    # Check available code families without loading data
    available_families = get_available_code_families()
    
    if not available_families:
        print("No code families found in data folder!")
        return None
    
    # Interactive selection of families
    selected_family_names = select_code_families_to_process(available_families)
    
    if not selected_family_names:
        print("No families selected for processing.")
        return None
    
    # Load only the selected families
    selected_codes = load_selected_code_families(selected_family_names, available_families)
    
    if not selected_codes:
        print("No codes loaded from selected families.")
        return None
    
    # Get number of processes
    import multiprocessing
    max_cores = multiprocessing.cpu_count()
    
    while True:
        try:
            n_processes_input = input(f"\nNumber of parallel processes (1-{max_cores}): ").strip()
            n_processes = int(n_processes_input)
            if 1 <= n_processes <= max_cores:
                break
            else:
                print(f"Error: Please enter a number between 1 and {max_cores}")
        except ValueError:
            print("Error: Please enter a valid integer")
    
    print(f"\nProcessing {len(selected_codes)} selected codes with {n_processes} processes...")
    
    # Process using parallel HAL
    results = process_codes_parallel(selected_codes, n_processes=n_processes)
    
    # Create enhanced visualization
    if results:
        families = create_visualization(results)
        
        # Export comprehensive summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'processing_mode': 'parallel',
            'total_codes_processed': len(results),
            'families': {family: len(data['x']) for family, data in families.items()},
            'performance_summary': {
                'avg_processing_time_per_code': np.mean([r['processing_time'] for r in results]),
                'total_processing_time': sum(r['processing_time'] for r in results),
                'parallelization_benefit': 'Estimated significant speedup vs sequential'
            }
        }
        
        with open('results/parallel_processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Comprehensive summary saved to results/parallel_processing_summary.json")
    
    return results


if __name__ == "__main__":
    results = main()