#!/usr/bin/env python3
import subprocess
import sys
import shutil
from pathlib import Path

# 기본 lowering 패스 세트
LOWERING_PASSES = [
    # Bufferization
    '--one-shot-bufferize=bufferize-function-boundaries',
    '--convert-bufferization-to-memref',

    # Linalg
    '--convert-linalg-to-loops',

    # Affine
    '--lower-affine',

    # Control flow
    '--convert-scf-to-cf',
    '--convert-cf-to-llvm',

    # Arithmetic / Math
    '--convert-arith-to-llvm',
    '--convert-math-to-llvm',

    # Func, Index
    '--convert-func-to-llvm',
    '--convert-index-to-llvm',

    # Memref
    '--expand-strided-metadata',
    '--memref-expand',
    '--finalize-memref-to-llvm',

    # Casts
    '--reconcile-unrealized-casts',
]

def run_cmd(cmd, input_file, output_file=None):
    """Helper to run shell commands with optional output redirection."""
    args = cmd + [str(input_file)]
    if output_file:
        with open(output_file, "w") as f:
            subprocess.check_call(args, stdout=f)
    else:
        subprocess.check_call(args)

def detect_dialects(mlir_file):
    """Return set of dialect names appearing in the MLIR file."""
    text = Path(mlir_file).read_text()
    dialects = set()
    for line in text.splitlines():
        if '"' in line or "." in line:
            if " " in line:
                token = line.strip().split()[0]
                if "." in token:
                    dialects.add(token.split(".")[0])
    return dialects

def main():
    if len(sys.argv) < 2:
        print("Usage: auto_lower.py <input.mlir> [--emit-ptx]")
        sys.exit(1)

    input_file = Path(sys.argv[1]).resolve()
    work_file = Path("work_lowered.mlir")
    emit_ptx = ("--emit-ptx" in sys.argv)

    # 반복 lowering
    dialects = detect_dialects(input_file)
    iter_count = 0
    while any(d not in ("builtin", "llvm", "func") for d in dialects):
        iter_count += 1
        print(f"[pass round {iter_count}] Dialects found: {dialects}")

        # mlir-opt 실행
        cmd = ["mlir-opt"] + LOWERING_PASSES + ["-o", str(work_file)]
        run_cmd(cmd, input_file)

        input_file = work_file
        dialects = detect_dialects(input_file)

        if iter_count > 10:
            print("Too many iterations, stopping.")
            sys.exit(1)

    # LLVM IR로 변환
    print("[translate] MLIR → LLVM IR")
    run_cmd(["mlir-translate", "--mlir-to-llvmir"], input_file, "output.ll")

    if emit_ptx:
        print("[llc] LLVM IR → PTX")
        run_cmd(["llc", "-march=nvptx64", "-mcpu=sm_75", "output.ll", "-o", "kernel.ptx"], input_file=None)

    print("Done. Results: output.ll", "+ kernel.ptx" if emit_ptx else "")

if __name__ == "__main__":
    main()

