"""
Command-line interface for QAero.
"""
import argparse
import sys
import json
import tempfile
import os
from typing import Optional

from ..core.solver import create_solver
from ..personas import create_persona
from ..core.metrics import global_benchmark_suite
from ..problems.aerodynamics import AirfoilOptimizationProblem
from ..problems.trajectory import AscentTrajectoryProblem


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QAero: Quantum Aerospace Optimization & Simulation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qaero init                          # Initialize QAero environment
  qaero run-benchmark                 # Run comprehensive benchmarks
  qaero analyze --problem airfoil     # Analyze airfoil optimization
  qaero solve --problem wing --backend quantum  # Solve wing design with quantum backend
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize QAero environment')
    init_parser.add_argument('--config', type=str, help='Configuration file path')
    
    # Run-benchmark command
    benchmark_parser = subparsers.add_parser('run-benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--benchmark', type=str, help='Specific benchmark to run')
    benchmark_parser.add_argument('--output', type=str, help='Output file for results')
    benchmark_parser.add_argument('--backend', type=str, default='auto', help='Backend to use')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze problems and results')
    analyze_parser.add_argument('--problem', type=str, required=True, help='Problem type to analyze')
    analyze_parser.add_argument('--input', type=str, help='Input results file')
    analyze_parser.add_argument('--output', type=str, help='Output analysis file')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve specific problem')
    solve_parser.add_argument('--problem', type=str, required=True, help='Problem type to solve')
    solve_parser.add_argument('--backend', type=str, default='auto', help='Backend to use')
    solve_parser.add_argument('--algorithm', type=str, help='Algorithm to use')
    solve_parser.add_argument('--output', type=str, help='Output results file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'init':
            init_environment(args)
        elif args.command == 'run-benchmark':
            run_benchmarks(args)
        elif args.command == 'analyze':
            analyze_problem(args)
        elif args.command == 'solve':
            solve_problem(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def init_environment(args):
    """Initialize QAero environment."""
    print("Initializing QAero environment...")
    
    # Check dependencies
    check_dependencies()
    
    # Create default configuration
    config = {
        'default_backend': 'classical_scipy',
        'auto_fallback': True,
        'visualization': True,
        'cache_results': True
    }
    
    config_path = args.config or 'qaero_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_path}")
    print("QAero environment initialized successfully!")


def run_benchmarks(args):
    """Run performance benchmarks."""
    print("Running QAero benchmarks...")
    
    solver = create_solver({'backend': args.backend})
    
    if args.benchmark:
        # Run specific benchmark
        result = global_benchmark_suite.run_single(args.benchmark, solver)
        print(f"Benchmark {args.benchmark} completed: success={result.success}")
    else:
        # Run all benchmarks
        results = global_benchmark_suite.run_all(solver)
        report = global_benchmark_suite.get_comparison_report()
        
        print("\nBenchmark Results Summary:")
        print(f"Total benchmarks: {report['total_benchmarks']}")
        print(f"Successful: {report['successful_benchmarks']}")
        print(f"Average execution time: {report['average_execution_time']:.2f}s")
    
    if args.output:
        global_benchmark_suite.save_results(args.output)
        print(f"Results saved to {args.output}")


def analyze_problem(args):
    """Analyze problem and results."""
    print(f"Analyzing {args.problem} problem...")
    
    if args.input:
        # Analyze existing results
        with open(args.input, 'r') as f:
            results_data = json.load(f)
        print(f"Loaded results from {args.input}")
        # Perform analysis...
    else:
        # Create and analyze problem
        persona = create_persona('aerospace_engineer')
        problem = persona.create_problem(args.problem)
        
        print(f"Problem: {problem.problem_id}")
        print(f"Variables: {problem.variables}")
        print(f"Parameters: {problem.parameters}")
        
        # Perform problem analysis...
    
    if args.output:
        analysis_results = {"analysis": "completed"}  # Placeholder
        with open(args.output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"Analysis saved to {args.output}")


def solve_problem(args):
    """Solve specific problem."""
    print(f"Solving {args.problem} problem with {args.backend} backend...")
    
    # Create problem
    persona = create_persona('aerospace_engineer')
    problem = persona.create_problem(args.problem)
    
    # Create solver
    solver_config = {'backend': args.backend}
    if args.algorithm:
        solver_config['algorithm'] = args.algorithm
    
    solver = create_solver(solver_config)
    
    # Solve problem
    result = solver.solve(problem)
    
    # Display results
    print(f"Solution completed: success={result.success}")
    if result.success:
        print(f"Optimal value: {result.optimal_value}")
        print(f"Execution time: {result.execution_time:.2f}s")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Results saved to {args.output}")


def check_dependencies():
    """Check required dependencies."""
    dependencies = ['numpy', 'scipy', 'matplotlib']
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            print(f"Warning: {dep} not installed. Some features may not work.")


if __name__ == '__main__':
    main()