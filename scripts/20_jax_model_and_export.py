# 10_jax_model_and_export.py  (Python 3.11)
import os, time, json, sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import export as jax_export
except Exception:
    jax_export = None

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
# StableHLO Export
# ─────────────────────────────────────────────────────────────────────────────
def export_stablehlo_prepartition(params_host, x_host_nhwc, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    def _spec_like(a):
        a = np.asarray(a)
        return jax.ShapeDtypeStruct(a.shape, a.dtype)

    params_spec = jax.tree_util.tree_map(_spec_like, params_host)
    x_spec = jax.ShapeDtypeStruct(np.shape(x_host_nhwc), np.asarray(x_host_nhwc).dtype)

    if jax_export is not None and hasattr(jax_export, "export"):
        exported = jax_export.export(jax.jit(forward))(params_spec, x_spec)
        mlir_mod = exported.mlir_module()
        (out_dir / "stablehlo").mkdir(exist_ok=True)
        with open(out_dir / "stablehlo" / "model.mlir", "w") as f:
            f.write(str(mlir_mod))
        print(f"[JAX/export] pre-partition StableHLO saved → {out_dir/'stablehlo'}")
    else:
        raise RuntimeError("jax.export API가 지원되지 않는 환경입니다.")

    np.savez(out_dir / "weights.npz",
             conv_w=params_host["conv_w"],
             conv_b=params_host["conv_b"],
             lin_w=params_host["lin_w"],
             lin_b=params_host["lin_b"])

    spec_meta = {k: {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)}
                 for k, v in params_host.items()}
    with open(out_dir / "params_spec.json", "w") as f:
        json.dump(spec_meta, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# Sharding 실행 (TP 예시)
# ─────────────────────────────────────────────────────────────────────────────
def build_mesh_2x2():
    devs = np.array(jax.devices()[:4])
    assert devs.size >= 4, f"필요 GPU=4, 감지된 디바이스={len(jax.devices())}"
    return Mesh(devs.reshape(2, 2), axis_names=("data", "model"))

def shard_like(a_np, mesh: Mesh, pspec):
    sharding = NamedSharding(mesh, pspec)
    return jax.device_put(a_np, sharding)

def run_tp(params_host, x_host_nhwc, iters=5):
    mesh = build_mesh_2x2()
    with mesh:
        x = shard_like(x_host_nhwc, mesh, P("data", None, None, None))
        params = {
            "conv_w": shard_like(params_host["conv_w"], mesh, P(None, None, None, "model")),
            "conv_b": shard_like(params_host["conv_b"], mesh, P("model",)),
            "lin_w" : shard_like(params_host["lin_w"],  mesh, P("model", None)),
            "lin_b" : shard_like(params_host["lin_b"],  mesh, P("model",)),
        }
        # prime
        y = forward_jit(params, x)
        y.block_until_ready()
        # measure
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = forward_jit(params, x)
            y.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        print("\n[DP+TP Benchmark]")
        print(f"  min   {min(times)*1000:.3f} ms")
        print(f"  max   {max(times)*1000:.3f} ms")
        print(f"  mean  {sum(times)/len(times)*1000:.3f} ms")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    params_host, x_host_nhwc = load_from_npz(RESULT_DIR / "weights_inputs.npz")

    # StableHLO export
    export_stablehlo_prepartition(params_host, x_host_nhwc, RESULT_DIR / "jax_pre_stablehlo")

    # TP 실행해서 성능 측정 + XLA dump 생성
    run_tp(params_host, x_host_nhwc, iters=10)

    print(f"\nHLO dumps → {dump_dir}")

