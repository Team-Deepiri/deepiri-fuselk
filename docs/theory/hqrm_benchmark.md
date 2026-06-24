# JAX HQRM Latency Benchmark

## Claim

> HQRM kernel inference **< 1 ms** mean latency on GPU (post-JIT warmup).

## Implementation

- `helix/jax_hqrm.py` — JIT-compiled 49-point kernel variance
- `helix/helical_quadtree.py` — full recursive NumPy HQRM (reference)

## Run benchmark

```bash
poetry install --with jax
fuselk doctor --vision
# or scripts/benchmark.py --vision
```

`HelixEngine.process()` calls `run_hqrm_jax()` on every frame.

## Report fields

| Field | Meaning |
|-------|---------|
| `mean_ms` | Mean latency after warmup |
| `p99_ms` | 99th percentile |
| `sub_ms_claim_met` | `mean_ms < 1.0` |
| `speedup_vs_numpy` | JAX vs NumPy HQRM |

## CPU fallback

Without JAX or on CPU-only hosts, NumPy HQRM is reported separately (target < 50 ms). Sub-ms claim requires GPU + JAX group.
