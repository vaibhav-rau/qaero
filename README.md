# QAero: Quantum Aerospace Optimization & Simulation Toolkit

[![CI](https://github.com/qaero-dev/qaero/actions/workflows/ci.yml/badge.svg)](https://github.com/qaero-dev/qaero/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/qaero)](https://pypi.org/project/qaero/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-grade quantum computing for aerospace engineering.**

QAero bridges the gap between theoretical quantum advantage and practical aerospace design workflows. We provide robust, intuitive interfaces for quantum-assisted optimization and quantum-accelerated simulation.

## 🚀 Features

- **Quantum Optimization**: QAOA, quantum annealing, variational algorithms for aerospace problems
- **PDE Solvers**: Quantum-accelerated CFD and structural mechanics
- **Problem Templates**: Pre-built aerospace problems (airfoils, wings, trajectories)
- **Hybrid Backends**: Seamless classical/quantum computation with automatic fallbacks
- **Production Ready**: Enterprise-grade testing, documentation, and security

## 📦 Installation

```bash
# Core installation
pip install qaero

# With quantum backends
pip install qaero[quantum]

# With D-Wave support
pip install qaero[dwave]

# Development installation
pip install qaero[dev]