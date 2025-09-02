# 10_pt_model_and_export.py  (Python 3.11)
import os, sys, time, copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from contextlib import contextmanager
# NVTX
from torch.cuda import nvtx
@contextmanager
def nvtx_range(msg: str):
    nvtx.range_push(msg)
    try:
        yield
    finally:
        nvtx.range_pop()

sys.path.append(str(Path(__file__).resolve().parent.parent))
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 환경변수: 반드시 torch/torch_xla import 전에 설정
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("PJRT_DEVICE", "CUDA")
os.environ.setdefault("GPU_NUM_DEVICES", "4")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

dump_dir = RESULT_DIR / "xla_dumps_pt"
dump_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_dump_to={dump_dir} "
    "--xla_dump_hlo_as_text=true "
    "--xla_dump_hlo_as_dot=true "
    "--xla_dump_include_timestamp=false "
    "--xla_dump_hlo_pass_re=.*"
)

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

# SPMD 모드는 XLA 텐서/모델 생성 전에 켜기
xr.use_spmd()

# ─────────────────────────────────────────────────────────────────────────────
# 모델 정의 (Conv2d → Flatten → Linear → ReLU)
#  - 실제 Conv/Linear의 크기 정의는 NPZ 로딩 시 재구성합니다.
#    (초기값은 placeholder이며, load_from_npz()에서 교체됨)
# ─────────────────────────────────────────────────────────────────────────────
class ConvLinearRelu(nn.Module):
    def __init__(self):
        super().__init__()
        # placeholder (NPZ 로딩 시 실제 크기로 교체)
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 30 * 30, 16, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.flatten(y)
        y = self.fc(y)
        return self.relu(y)

# ─────────────────────────────────────────────────────────────────────────────
# 공통: 동일 입력/가중치 로드(npz) + 모델을 NPZ shape에 맞게 재구성
# ─────────────────────────────────────────────────────────────────────────────
def _out_len_after_conv2d(h: int, w: int, kh: int, kw: int, stride=1, padding=0, dilation=1):
    # PyTorch Conv2d 출력 크기 계산식
    oh = (h + 2*padding - dilation*(kh - 1) - 1) // stride + 1
    ow = (w + 2*padding - dilation*(kw - 1) - 1) // stride + 1
    return oh, ow

def load_from_npz(m: nn.Module, path=f"{RESULT_DIR}/weights_inputs.npz"):
    """
    NPZ의 배열 shape를 읽어 모델을 해당 크기로 재구성하고, 가중치/입력을 로드합니다.
    - x      : (N, C, H, W)
    - w_conv : (OC, IC, KH, KW)
    - b_conv : (OC,)
    - w_lin  : (OUT, in_lin=OC*OH*OW)  # OH,OW는 stride=1, padding=0 가정
    - b_lin  : (OUT,)
    """
    z = np.load(path)
    x_np = z["x"]
    w_conv_np = z["w_conv"]; b_conv_np = z["b_conv"]
    w_lin_np  = z["w_lin"];  b_lin_np  = z["b_lin"]

    # shapes
    N, C, H, W = x_np.shape
    OC, IC, KH, KW = w_conv_np.shape
    OUT, in_lin = w_lin_np.shape

    # 일관성 체크
    assert C == IC, f"입력 C({C})와 Conv IC({IC})가 다릅니다."
    OH, OW = _out_len_after_conv2d(H, W, KH, KW, stride=1, padding=0, dilation=1)
    calc_in_lin = OC * OH * OW
    assert in_lin == calc_in_lin, f"Linear in_features({in_lin}) != OC*OH*OW({calc_in_lin})"
    assert b_conv_np.shape == (OC,), "b_conv shape 불일치"
    assert b_lin_np.shape  == (OUT,), "b_lin shape 불일치"

    # 모델을 해당 크기로 재구성 (CPU 상에서)
    with torch.no_grad():
        m.conv = nn.Conv2d(
            in_channels=C, out_channels=OC,
            kernel_size=(KH, KW), stride=1, padding=0, bias=True
        )
        m.fc = nn.Linear(in_features=in_lin, out_features=OUT, bias=True)

        # 가중치 로드
        m.conv.weight.copy_(torch.from_numpy(w_conv_np))
        m.conv.bias.copy_(torch.from_numpy(b_conv_np))
        m.fc.weight.copy_(torch.from_numpy(w_lin_np))
        m.fc.bias.copy_(torch.from_numpy(b_lin_np))

    # 입력 텐서
    x = torch.from_numpy(x_np.astype(np.float32))
    m.eval()

    # 정보 출력
    print(f"[Shapes] x={tuple(x.shape)}, conv_w={tuple(w_conv_np.shape)}, fc_w={tuple(w_lin_np.shape)}")
    print(f"[Derived] OHxOW={OH}x{OW}, in_lin={in_lin}, OUT={OUT}")
    return m, x

# ─────────────────────────────────────────────────────────────────────────────
# StableHLO(사전 파티셔닝) export
# ─────────────────────────────────────────────────────────────────────────────
def export_stablehlo_prepartition(m: nn.Module, x_cpu: torch.Tensor, out_dir: Path):
    ep = export(m, (x_cpu,))
    shlo = exported_program_to_stablehlo(ep)
    shlo.save(out_dir)  # 디렉터리(prefix)로 저장됨
    print(f"[PyTorch] pre-partition StableHLO saved → {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# 벤치마크 유틸
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh_2x2():
    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 4, f"필요 GPU=4, 감지된 디바이스={num_devices}"
    device_ids = np.arange(4)
    mesh_shape = (2, 2)
    axis_names = ('data', 'model')
    return Mesh(device_ids, mesh_shape, axis_names)

def to_xla(m: nn.Module, x_cpu: torch.Tensor):
    dev = xm.xla_device()
    m = m.to(dev)
    x = x_cpu.to(dev)
    # 호스트→디바이스 업로드를 여기서 끝내고, 이후 측정에서 제외
    xm.mark_step(); xm.wait_device_ops()
    return m, x

def clear_all_sharding(m: nn.Module, x: torch.Tensor):
    try:
        xs.clear_sharding(x)
    except Exception:
        pass
    for p in m.parameters():
        try:
            xs.clear_sharding(p)  # 파라미터에 남은 주석 제거
        except Exception:
            pass

def prime_once(m: nn.Module, x: torch.Tensor, label: str | None = None):
    with torch.no_grad():
        with nvtx_range(f"Prime: {label}"):
            _ = m(x)
            xm.mark_step()
            xm.wait_device_ops()

def measure_iters(m: nn.Module, x: torch.Tensor, warmup=10, iters=10, nvtx_prefix: str = ""):
    # Warmup: 초기 실행(데이터 분배/초기화) 후 측정
    for i in range(warmup):
        with torch.no_grad():
            with nvtx_range(f"{nvtx_prefix}/Warmup/{i}"):
                _ = m(x)
                xm.mark_step()
                xm.wait_device_ops()
    # 측정
    capture_ctx = nvtx_range(f"nsys capture: {nvtx_prefix}")
    with capture_ctx:
        times = []
        for _ in range(iters):
            with torch.no_grad():
                t0 = time.perf_counter()
                _ = m(x)
                xm.mark_step()
                xm.wait_device_ops()
                t1 = time.perf_counter()
            times.append(t1 - t0)
    return times

def summarize_times(label: str, times: list[float]):
    ms = [t * 1000.0 for t in times]
    print(f"\n[{label}] 10회 측정(ms)")
    print(f"  min    : {min(ms):9.3f}")
    print(f"  max    : {max(ms):9.3f}")
    print(f"  mean   : {sum(ms)/len(ms):9.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 케이스 A: 데이터 병렬(나이브) — 입력 배치만 분할, 파라미터는 복제
# ─────────────────────────────────────────────────────────────────────────────
def bench_dp_only(m: nn.Module, x_cpu: torch.Tensor):
    mesh = build_mesh_2x2()
    m, x = to_xla(copy.deepcopy(m), x_cpu)
    clear_all_sharding(m, x)

    # 입력 배치(N)만 'data' 축으로 샤딩 -> 파라미터는 미주석(복제)
    xs.mark_sharding(x, mesh, ('data', None, None, None))

    # 프라임 1회 (데이터 분배/초기 실행)
    prime_once(m, x, label="DP")

    times = measure_iters(m, x, warmup=10, iters=10, nvtx_prefix="DP")
    summarize_times("Naive DP (batch만 분할, 파라미터 복제)", times)
    return times

# ─────────────────────────────────────────────────────────────────────────────
# 케이스 B: 데이터+모델 병렬(GSPMD) — 입력은 'data', 파라미터는 'model' 축으로 분할
# ─────────────────────────────────────────────────────────────────────────────
def bench_dp_tp(m: nn.Module, x_cpu: torch.Tensor):
    mesh = build_mesh_2x2()
    m, x = to_xla(copy.deepcopy(m), x_cpu)
    clear_all_sharding(m, x)

    # 입력 배치 샤딩
    xs.mark_sharding(x, mesh, ('data', None, None, None))
    # 파라미터 모델 샤딩(OC/out_features 분할)
    xs.mark_sharding(m.conv.weight, mesh, ('model', None, None, None))
    xs.mark_sharding(m.conv.bias,   mesh, ('model',))
    xs.mark_sharding(m.fc.weight,   mesh, ('model', None))
    xs.mark_sharding(m.fc.bias,     mesh, ('model',))

    # 프라임 1회
    prime_once(m, x, label="DP+TP")

    times = measure_iters(m, x, warmup=10, iters=10, nvtx_prefix="DP+TP")
    summarize_times("DP+TP (데이터+모델 축 분할, GSPMD)", times)
    return times

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_m = ConvLinearRelu()
    base_m, x_cpu = load_from_npz(base_m, path=f"{RESULT_DIR}/weights_inputs.npz")

    # 사전 파티셔닝 StableHLO 저장(샤딩 주석 없음)
    export_stablehlo_prepartition(base_m, x_cpu, RESULT_DIR / "pt_pre_stablehlo")

    # 벤치마크: (A) 나이브 데이터 병렬 vs (B) 데이터+모델 병렬
    dp_times = bench_dp_only(base_m, x_cpu)
    dptp_times = bench_dp_tp(base_m, x_cpu)

    # 간단 비교 요약
    def stats(ts):
        return min(ts), max(ts), sum(ts)/len(ts)
    a_min, a_max, a_mean = stats(dp_times)
    b_min, b_max, b_mean = stats(dptp_times)

    print("\n=== 요약 (초) ===")
    print(f"Naive DP        -> min {a_min:.6f}, max {a_max:.6f}, mean {a_mean:.6f}")
    print(f"DP+TP (GSPMD)   -> min {b_min:.6f}, max {b_max:.6f}, mean {b_mean:.6f}")
    print(f"\nHLO dumps → {dump_dir}")

