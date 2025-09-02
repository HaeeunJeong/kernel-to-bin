# kernel-to-bin

`kernel-to-bin` is an experimental project for generating executable binaries (PTX, CUBIN, etc.) from kernel IR using an MLIR/LLVM-based workflow.  
It explores the lowering path from PyTorch or JAX exported StableHLO → Linalg → GPU dialect → LLVM → PTX.

---

## Features

- **StableHLO → Linalg conversion**: Lowering via `stablehlo-opt`
- **Linalg → GPU dialect conversion**: Using `mlir-opt` passes
- **GPU dialect → NVVM/LLVM IR conversion**: CUDA backend codegen
- **PTX / CUBIN generation**: Final kernel binaries with `llc` and `ptxas`
- **Experimental scripts**: Python utilities for end-to-end export and transformation

---

## Requirements

- Python 3.11+
- PyTorch (with `torch_xla`)
- JAX (optional, for StableHLO export experiments)
- MLIR/LLVM (must be built from source or installed)
- CUDA Toolkit (e.g., CUDA 12.x, including `ptxas`)

---

## Getting Started

```bash
git clone https://github.com/HaeeunJeong/kernel-to-bin.git
conda env create -f env/environment.yaml
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
````

---

## Usage

1. Export a PyTorch model to StableHLO:
```bash
python scripts/00_make_artifacts.py
python scripts/10_pt_model_and_export.py
````

2. Convert StableHLO to Linalg to LLVM IR:

```bash
python 30_run_pipeline.py
```

3. Lower Linalg to LLVM IR, then generate PTX:

```bash
cd results/pipeline

mlir-translate --mlir-to-llvmir results/pipeline/llvm.mlir > results/pipeline/output.ll
llc -march=nvptx64 -mcpu=sm_75 results/pipeline/output.ll -o results/pipeline/kernel.ptx
ptxas -arch=sm_75 results/pipeline/kernel.ptx -o results/pipeline/kernel.cubin

nvcc -o utils/run_kernel utils/run_kernel.cpp -lcuda
./run_kernel

```
