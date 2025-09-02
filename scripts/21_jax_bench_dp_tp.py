# 11_jax_bench_dp_tp.py (Python 3.11)
import time, os, sys, json
import numpy as np
from pathlib import Path
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from torch.cuda import nvtx
from contextlib import contextmanager

# ─────────────────────────────────────────────────────────────────────────────
# 환경 변수 설정
# ─────────────────────────────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

dump_dir = RESULT_DIR / "xla_dumps_jax"
dump_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_dump_to={dump_dir} "
    "--xla_dump_hlo_as_text=true "
    "--xla_dump_hlo_as_dot=true "
    "--xla_dump_include_timestamp=false "
    "--xla_dump_hlo_pass_re=.*"
)




# ─────────────────────────────────────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────────────────────────────────────
def forward(params, x_nhwc):
    y = lax.conv_general_dilated(
        x_nhwc,
        params["conv_w"],
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=1,
    )
    y = y + params["conv_b"]
    y = y.reshape((y.shape[0], -1))
    y = jnp.matmul(y, params["lin_w"].T) + params["lin_b"]
    return jnp.maximum(y, 0.0)

forward_jit = jax.jit(forward)

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로더
# ─────────────────────────────────────────────────────────────────────────────
def load_from_npz(path=RESULT_DIR / "weights_inputs.npz"):
    z = np.load(path)
    x_nchw = z["x"].astype(np.float32)
    w_oi_hw = z["w_conv"].astype(np.float32)
    b_conv = z["b_conv"].astype(np.float32)
    w_lin = z["w_lin"].astype(np.float32)
    b_lin = z["b_lin"].astype(np.float32)

    # NHWC/HWIO 변환
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1))
    w_hwio = np.transpose(w_oi_hw, (2, 3, 1, 0))

    params = {
        "conv_w": w_hwio,
        "conv_b": b_conv,
        "lin_w": w_lin,
        "lin_b": b_lin,
    }
    return params, x_nhwc

# ─────────────────────────────────────────────────────────────────────────────
# NVTX range helper
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def nvtx_range(msg: str):
    nvtx.range_push(msg)
    try:
        yield
    finally:
        nvtx.range_pop()

# ─────────────────────────────────────────────────────────────────────────────
# Mesh & shard utils
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh_2x2():
    devs = np.array(jax.devices()[:4])
    assert devs.size >= 4, f"필요 GPU=4, 감지된 디바이스={len(jax.devices())}"
    mesh = Mesh(devs.reshape(2, 2), axis_names=("data", "model"))
    return mesh

def shard_like(a_np, mesh: Mesh, pspec):
    sharding = NamedSharding(mesh, pspec)
    return jax.device_put(a_np, sharding)

# ─────────────────────────────────────────────────────────────────────────────
# Bench utils
# ─────────────────────────────────────────────────────────────────────────────
def prime_once(fn, params, x, label=None):
    with nvtx_range(f"Prime: {label}" if label else "Prime"):
        y = fn(params, x)
        y.block_until_ready()

def measure_iters(fn, params, x, warmup=10, iters=10, nvtx_prefix=""):
    for i in range(warmup):
        with nvtx_range(f"{nvtx_prefix}/Warmup/{i}"):
            y = fn(params, x)
            y.block_until_ready()
    times = []
    with nvtx_range(f"nsys capture: {nvtx_prefix}"):
        for _ in range(iters):
            t0 = time.perf_counter()
            y = fn(params, x)
            y.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return times

def summarize_times(label, times_s):
    ms = [t * 1000 for t in times_s]
    print(f"\n[{label}] 10회 측정(ms)")
    print(f"  min    : {min(ms):9.3f}")
    print(f"  max    : {max(ms):9.3f}")
    print(f"  mean   : {sum(ms)/len(ms):9.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Bench cases
# ─────────────────────────────────────────────────────────────────────────────
def bench_dp_only(params_host, x_host_nhwc):
    mesh = build_mesh_2x2()
    with mesh:
        x = shard_like(x_host_nhwc, mesh, P("data", None, None, None))
        params = {
            "conv_w": shard_like(params_host["conv_w"], mesh, P()),
            "conv_b": shard_like(params_host["conv_b"], mesh, P()),
            "lin_w" : shard_like(params_host["lin_w"],  mesh, P()),
            "lin_b" : shard_like(params_host["lin_b"],  mesh, P()),
        }
        prime_once(forward_jit, params, x, label="DP")
        times = measure_iters(forward_jit, params, x, warmup=10, iters=10, nvtx_prefix="DP")
        summarize_times("Naive DP (batch만 분할, 파라미터 복제)", times)
        return times

def bench_dp_tp(params_host, x_host_nhwc):
    mesh = build_mesh_2x2()
    with mesh:
        x = shard_like(x_host_nhwc, mesh, P("data", None, None, None))
        params = {
            "conv_w": shard_like(params_host["conv_w"], mesh, P(None, None, None, "model")),
            "conv_b": shard_like(params_host["conv_b"], mesh, P("model",)),
            "lin_w" : shard_like(params_host["lin_w"],  mesh, P("model", None)),
            "lin_b" : shard_like(params_host["lin_b"],  mesh, P("model",)),
        }
        prime_once(forward_jit, params, x, label="DP+TP")
        times = measure_iters(forward_jit, params, x, warmup=10, iters=10, nvtx_prefix="DP+TP")
        summarize_times("DP+TP (데이터+모델 축 분할, Sharding)", times)
        return times

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    params_host, x_host_nhwc = load_from_npz(RESULT_DIR / "weights_inputs.npz")
    dp_times = bench_dp_only(params_host, x_host_nhwc)
    dptp_times = bench_dp_tp(params_host, x_host_nhwc)

    def stats(ts): return (min(ts), max(ts), sum(ts)/len(ts))
    a_min, a_max, a_mean = stats(dp_times)
    b_min, b_max, b_mean = stats(dptp_times)

    print("\n=== 요약 (초) ===")
    print(f"Naive DP        -> min {a_min:.6f}, max {a_max:.6f}, mean {a_mean:.6f}")
    print(f"DP+TP (Sharding)-> min {b_min:.6f}, max {b_max:.6f}, mean {b_mean:.6f}")

