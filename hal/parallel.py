#!/usr/bin/env python3
"""
Parallel processing utilities for HAL algorithm.
Enables batch processing of multiple QECC codes with multiprocessing.
"""

from multiprocessing import Pool, cpu_count
from functools import partial
import time
from typing import List, Dict, Any, Optional, Callable
from .hal import HAL
from .data_structures import QECCLayout
import networkx as nx


def process_single_code_worker(code_data: tuple) -> Dict[str, Any]:
    """
    Worker function to process a single code through HAL.
    Designed for parallel execution in multiprocessing.
    
    Args:
        code_data: Tuple of (code_info, graph_creator_func, index, total)
        
    Returns:
        Dictionary with processing results
    """
    try:
        code_info, graph_creator_func, index, total = code_data
        
        # Create HAL instance (each worker needs its own)
        hal = HAL()
        
        # Create graph using the provided function
        graph = graph_creator_func(code_info)
        
        if graph is None:
            return {
                'index': index,
                'success': False,
                'error': 'Could not create graph',
                'code_info': code_info
            }
            
        # Process through HAL
        start_time = time.time()
        layout = hal.layout_code(graph)
        processing_time = time.time() - start_time
        
        # Return comprehensive results
        return {
            'index': index,
            'success': True,
            'code_info': code_info,
            'layout': layout,
            'processing_time': processing_time,
            'graph_stats': {
                'nodes': len(graph.nodes()),
                'edges': len(graph.edges()),
                'avg_degree': sum(dict(graph.degree()).values()) / len(graph.nodes()) if len(graph.nodes()) > 0 else 0
            }
        }
        
    except Exception as e:
        return {
            'index': index,
            'success': False,
            'error': str(e),
            'code_info': code_info
        }


class ParallelHAL:
    """
    Parallel processing wrapper for HAL algorithm.
    Enables efficient batch processing of multiple QECC codes.
    """
    
    def __init__(self, n_processes: Optional[int] = None, verbose: bool = True):
        """
        Initialize parallel HAL processor.
        
        Args:
            n_processes: Number of parallel processes (default: CPU count)
            verbose: Enable progress reporting
        """
        self.n_processes = n_processes or cpu_count()
        self.verbose = verbose
        
    def process_batch(self, 
                     codes: List[Dict[str, Any]], 
                     graph_creator: Callable,
                     chunk_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple codes in parallel through HAL.
        
        Args:
            codes: List of code information dictionaries
            graph_creator: Function to create NetworkX graph from code info
            chunk_size: Size of work chunks for multiprocessing
            
        Returns:
            List of processing results
        """
        if self.verbose:
            print(f"Processing {len(codes)} codes using {self.n_processes} processes...")
        
        # Prepare work items
        work_items = [
            (code, graph_creator, i, len(codes)) 
            for i, code in enumerate(codes)
        ]
        
        # Calculate optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(codes) // (self.n_processes * 4))
        
        start_time = time.time()
        results = []
        
        # Process in parallel
        with Pool(processes=self.n_processes) as pool:
            if self.verbose:
                print(f"Starting parallel processing with chunk size {chunk_size}...")
            
            # Process all items
            results = pool.map(process_single_code_worker, work_items, chunksize=chunk_size)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if r['success']]
        
        if self.verbose:
            print(f"Parallel processing completed in {total_time:.2f}s")
            print(f"Successfully processed: {len(successful_results)}/{len(codes)} codes")
            print(f"Average time per code: {total_time/len(codes):.3f}s")
        
        return results
    
    def process_qecc_families(self, 
                             qecc_families: Dict[str, List[Dict]], 
                             graph_creators: Dict[str, Callable]) -> Dict[str, List[Dict]]:
        """
        Process multiple QECC code families in parallel.
        
        Args:
            qecc_families: Dict mapping family names to code lists
            graph_creators: Dict mapping family names to graph creation functions
            
        Returns:
            Dict mapping family names to processing results
        """
        all_results = {}
        
        for family_name, codes in qecc_families.items():
            if family_name not in graph_creators:
                print(f"Warning: No graph creator for {family_name}, skipping...")
                continue
                
            if self.verbose:
                print(f"\nProcessing {family_name} codes ({len(codes)} codes)...")
            
            family_results = self.process_batch(codes, graph_creators[family_name])
            all_results[family_name] = family_results
        
        return all_results
    
    def get_processing_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate processing statistics from results.
        
        Args:
            results: List of processing results
            
        Returns:
            Dictionary with comprehensive statistics
        """
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if not successful_results:
            return {
                'total_codes': len(results),
                'successful': 0,
                'failed': len(failed_results),
                'success_rate': 0.0
            }
        
        processing_times = [r['processing_time'] for r in successful_results]
        hardware_costs = [r['layout'].hardware_cost for r in successful_results]
        
        # Calculate metrics statistics
        metrics_stats = {}
        if successful_results:
            first_metrics = successful_results[0]['layout'].metrics
            for metric_name in first_metrics.keys():
                metric_values = [r['layout'].metrics.get(metric_name, 0) for r in successful_results]
                metrics_stats[metric_name] = {
                    'mean': sum(metric_values) / len(metric_values),
                    'min': min(metric_values),
                    'max': max(metric_values)
                }
        
        return {
            'total_codes': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results),
            'processing_time': {
                'total': sum(processing_times),
                'mean': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times)
            },
            'hardware_cost': {
                'mean': sum(hardware_costs) / len(hardware_costs),
                'min': min(hardware_costs),
                'max': max(hardware_costs)
            },
            'metrics_stats': metrics_stats,
            'failed_codes': [r['code_info'] for r in failed_results]
        }


def create_qecc_graph_from_info(code_info: Dict[str, Any]) -> Optional[nx.Graph]:
    """
    Default graph creator that works with code info containing family and [n,k,d].
    
    Args:
        code_info: Dictionary with 'family', 'n', 'k', 'd' keys
        
    Returns:
        NetworkX graph or None if creation fails
    """
    try:
        from .graph_utils import create_bicycle_code_graph, create_tile_code_graph, create_radial_code_graph
        
        family = code_info['family']
        n, k, d = code_info['n'], code_info['k'], code_info['d']
        
        if family == 'BB code':
            return create_bicycle_code_graph(n, k, d)
        elif family == 'Tile code':
            return create_tile_code_graph(n, k, d)
        elif family == 'Radial code':
            return create_radial_code_graph(n, k, d)
        else:
            raise ValueError(f"Unknown code family: {family}")
            
    except Exception as e:
        print(f"Warning: Could not create graph for {code_info}: {e}")
        return None