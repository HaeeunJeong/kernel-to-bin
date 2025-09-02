# 10_pt_model_and_export.py  (Python 3.11)
import os, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

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
    # "--xla_dump_hlo_as_html=true "
    # "--xla_dump_hlo_as_long_text=false "
    # "--xla_dump_hlo_as_text=true "
    # "--xla_dump_hlo_as_dot=true "
    "--xla_dump_hlo_snapshots=true "
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
# ─────────────────────────────────────────────────────────────────────────────
class ConvLinearRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True)  # 32→30
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 30 * 30, 16, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.flatten(y)
        y = self.fc(y)
        return self.relu(y)

# ─────────────────────────────────────────────────────────────────────────────
# 공통: 동일 입력/가중치 로드(npz)
# ─────────────────────────────────────────────────────────────────────────────
def load_from_npz(m: nn.Module, path=f"{RESULT_DIR}/weights_inputs.npz"):
    z = np.load(path)
    with torch.no_grad():
        m.conv.weight.copy_(torch.from_numpy(z["w_conv"]))
        m.conv.bias.copy_(torch.from_numpy(z["b_conv"]))
        m.fc.weight.copy_(torch.from_numpy(z["w_lin"]))
        m.fc.bias.copy_(torch.from_numpy(z["b_lin"]))
    x = torch.from_numpy(z["x"])  # NCHW
    return m.eval(), x

# ─────────────────────────────────────────────────────────────────────────────
# StableHLO(사전 파티셔닝) export
# ─────────────────────────────────────────────────────────────────────────────
def export_stablehlo_prepartition(m: nn.Module, x_cpu: torch.Tensor, out_dir: Path):
    ep = export(m, (x_cpu,))
    shlo = exported_program_to_stablehlo(ep)
    shlo.save(out_dir)  # 디렉터리(prefix)로 저장됨
    print(f"[PyTorch] pre-partition StableHLO saved → {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# SPMD(GSPMD) 실행: 2x2 mesh = ('data','model')
#  - 입력 x: ('data', None, None, None)
#  - conv.weight: ('model', None, None, None)
#  - conv.bias:   ('model',)
#  - fc.weight:   ('model', None)
#  - fc.bias:     ('model',)
#  (출력 y는 전파로 샤딩이 정해지므로 별도 mark 불필요)
# ─────────────────────────────────────────────────────────────────────────────
def run_spmd_and_dump(m: nn.Module, x_cpu: torch.Tensor):
    dev = xm.xla_device()
    m = m.to(dev)
    x = x_cpu.to(dev)

    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 4, f"필요 GPU=4, 감지된 디바이스={num_devices}"
    device_ids = np.arange(4)
    mesh_shape = (2, 2)
    axis_names = ('data', 'model')
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    # 입력 배치 샤딩
    xs.mark_sharding(x, mesh, ('data', None, None, None))
    # 파라미터 모델 샤딩
    xs.mark_sharding(m.conv.weight, mesh, ('model', None, None, None))
    xs.mark_sharding(m.conv.bias,   mesh, ('model',))
    xs.mark_sharding(m.fc.weight,   mesh, ('model', None))
    xs.mark_sharding(m.fc.bias,     mesh, ('model',))

    # 실행(컴파일 트리거) + 동기화
    with torch.no_grad():
        y = m(x)
        _ = y.sum().cpu().item()
    xm.mark_step()
    xm.wait_device_ops()

    print(f"[PyTorch] ran with 2x2 Mesh('data','model'); dumps → {dump_dir}")
    return y.cpu()

# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = ConvLinearRelu()
    m, x_cpu = load_from_npz(m, path=f"{RESULT_DIR}/weights_inputs.npz")

    export_stablehlo_prepartition(m, x_cpu, RESULT_DIR / "pt_pre_stablehlo")

    _ = run_spmd_and_dump(m, x_cpu)

