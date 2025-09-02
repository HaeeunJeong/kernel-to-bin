# 20_run_pipeline.py  (Python 3.11)
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_DIR = RESULT_DIR / "pipeline"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

# 원본 PyTorch export & run 스크립트
PT_SCRIPT = ROOT_DIR / "scripts" / "10_pt_model_and_export.py"

# 중간 산출물
STABLEHLO_DIR = RESULT_DIR / "pt_pre_stablehlo" / "functions"
STABLEHLO_FILE = STABLEHLO_DIR / "forward.mlir"   # torch-xla가 저장한 파일 이름 기준 수정 필요

LINALG_FILE = PIPELINE_DIR / "linalg.mlir"
LLVM_FILE = PIPELINE_DIR / "llvm.mlir"

def run_cmd(cmd, cwd=None):
    print(f"\n[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # 1) dump (StableHLO export)
    if mode in ("all", "dump"):
        run_cmd([sys.executable, str(PT_SCRIPT), "dump"])

    # 2) stablehlo → linalg
    if mode in ("all", "dump"):
        run_cmd([
            "stablehlo-opt",
            "--stablehlo-legalize-to-linalg",
            str(STABLEHLO_FILE),
            "-o", str(LINALG_FILE)
        ])

    # 3) linalg → llvm
    if mode in ("all", "dump"):
        run_cmd([
            "mlir-opt",
            # "--one-shot-bufferize",
            "--one-shot-bufferize=bufferize-function-boundaries", 
            "--convert-bufferization-to-memref",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
            "--convert-func-to-llvm",
            "--convert-index-to-llvm",
            "--expand-strided-metadata",
            "--memref-expand",
            "--finalize-memref-to-llvm",
            "--reconcile-unrealized-casts",
            str(LINALG_FILE),
            "-o", str(LLVM_FILE)
        ])




    # 4) 실행 (SPMD run)
    if mode in ("all", "run"):
        run_cmd([sys.executable, str(PT_SCRIPT), "run"])

if __name__ == "__main__":
    main()

