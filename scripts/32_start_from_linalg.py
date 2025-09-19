#!/usr/bin/env python3
# Python 3.11
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
PIPELINE_DIR = RESULT_DIR / "pipeline"
DUMP_DIR = PIPELINE_DIR / "dump"

PTX_LOADER_SRC = ROOT_DIR / "utils" / "run_kernel.cpp"
PTX_LOADER_EXE = PIPELINE_DIR / "kernel"

def run_cmd(cmd, cwd=None):
    print(f"\n[CMD] {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True, cwd=cwd)

def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    DUMP_DIR.mkdir(parents=True, exist_ok=True)


def mlir_stage(in_file: Path, out_file: Path, passes: list[str]):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["mlir-opt", *passes, str(in_file), "-o", str(out_file)])

def linalg_to_llvm_with_dumps(linalg_in: Path) -> Path:
    """여러 단계로 나누어 중간 MLIR들을 dump 디렉터리에 저장한 뒤 최종 LLVM Dialect MLIR 경로 반환"""
    s0 = DUMP_DIR / "00_input_linalg.mlir"
    if linalg_in.resolve() != s0.resolve():
        shutil.copyfile(linalg_in, s0)

    # 1) bufferize
    s1 = DUMP_DIR / "10_bufferize.mlir"
    mlir_stage(
        s0,
        s1,
        ["--one-shot-bufferize=bufferize-function-boundaries", "--convert-bufferization-to-memref"],
    )

    # 2) linalg → loops, affine 낮춤
    s2 = DUMP_DIR / "20_linalg_to_loops_affine_lowered.mlir"
    mlir_stage(
        s1,
        s2,
        ["--convert-linalg-to-loops", "--lower-affine"],
    )

    # 3) scf → cf
    s3 = DUMP_DIR / "30_scf_to_cf.mlir"
    mlir_stage(
        s2,
        s3,
        ["--convert-scf-to-cf"],
    )

    # 4) LLVM dialect로 마무리
    s4 = DUMP_DIR / "40_to_llvm_dialect.mlir"
    mlir_stage(
        s3,
        s4,
        [
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-func-to-llvm",
            "--convert-index-to-llvm",
            "--expand-strided-metadata",
            "--memref-expand",
            "--finalize-memref-to-llvm",
            "--reconcile-unrealized-casts",
        ],
    )

    # 최종 산출물 복사(관례 파일명)
    final_llvm_mlir = PIPELINE_DIR / "llvm.mlir"
    shutil.copyfile(s4, final_llvm_mlir)
    return final_llvm_mlir

def llvm_mlir_to_ll(llvm_mlir_in: Path) -> Path:
    ll_out = PIPELINE_DIR / "module.ll"
    run_cmd([
        "mlir-translate",
        "--mlir-to-llvmir",
        str(llvm_mlir_in),
        "-o", str(ll_out)
    ])
    return ll_out

def ll_to_ptx(ll_in: Path) -> Path:
    ptx_out = PIPELINE_DIR / "kernel.ptx"
    run_cmd([
        "llc",
        "-O3",
        str(ll_in),
        "-o", str(ptx_out)
    ])
    return ptx_out

def build_ptx_loader() -> Path:
    # PTX_LOADER_EXE.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["clang++", "-std=c++17", str(PTX_LOADER_SRC), "-lcuda", "-o", str(PTX_LOADER_EXE)])
    return PTX_LOADER_EXE

def run_ptx(ptx_path: Path, kernel: str):
    exe = PTX_LOADER_EXE if PTX_LOADER_EXE.exists() else build_ptx_loader()
    run_cmd([str(exe), str(ptx_path), kernel])

def main():
    parser = argparse.ArgumentParser(
        description="Linalg → LLVM Dialect MLIR → LLVM IR → PTX → 실행 (중간산출물 dump 저장)"
    )
    parser.add_argument("--linalg", type=Path, required=True, help="입력 Linalg MLIR 파일 경로")
    parser.add_argument("--kernel", default="my_kernel", help="실행할 커널 심볼명")
    args = parser.parse_args()

    ensure_dirs()

    # 1) Linalg → LLVM Dialect MLIR, 중간 dump 저장
    llvm_mlir = linalg_to_llvm_with_dumps(args.linalg)

    # 2) LLVM Dialect MLIR → LLVM IR
    ll = llvm_mlir_to_ll(llvm_mlir)

    # 3) LLVM IR → PTX
    ptx = ll_to_ptx(ll)

    # 4) 실행
    run_ptx(ptx, args.kernel)

if __name__ == "__main__":
    main()

