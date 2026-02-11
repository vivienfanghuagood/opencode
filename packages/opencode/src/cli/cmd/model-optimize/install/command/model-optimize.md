---
description: "End-to-end vLLM model optimization pipeline. Usage: /model-optimize <model_name> [output_dir]"
agent: model-opt
---

# End-to-End vLLM Model Optimization Pipeline

## Target
- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `./model_opt_<model_short_name>`)

## First Steps
1. Parse model name from `$1` (e.g., "Qwen/Qwen3-8B" → short name "Qwen3-8B")
2. Determine output directory: `$2` if provided, else `./model_opt_<short_name>`
3. Create directory structure: `model/ demo/ profile/ problems/ optimized/ report/ scripts/`
4. Create `config.json` and `progress.json`
5. Copy helper scripts from `~/.config/opencode/scripts/` to `<output_dir>/scripts/`

## ⚠️ CRITICAL RULES
- **ALWAYS activate venv**: `source <output_dir>/venv/bin/activate`
- **NEVER modify system packages** in /opt/, /usr/
- **ALL decisions MUST be data-driven** — profile first, optimize second
- **Update progress.json after each phase**

## Helper Scripts (installed at `~/.config/opencode/scripts/`)
- `kernel_test_runner.py` — test kernel accuracy + benchmark
- `kernel_finalize.py` — save best optimization result
- `shape_capture.py` — capture dynamic shapes during inference
- `analyze_fusion.py` — detect operator fusion opportunities
- `vllm_trace_extractor.py` — extract GPU kernels from torch profiler trace
- `vllm_benchmark.py` — orchestrate vllm serve + bench serve
- `generate_vllm_plugin.py` — generate vLLM CustomOp plugin from optimized kernels

Copy them to the project at start:
```bash
SCRIPTS_SRC="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/scripts"
cp "$SCRIPTS_SRC"/*.py <output_dir>/scripts/
```

---

# Phase 0: Environment Setup 

## Goal
Create an isolated Python virtual environment with vLLM-rocm and all required dependencies.

## Steps

### 1. Detect ROCm Version
```bash
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "6.0")
echo "Detected ROCm version: $ROCM_VERSION"
```

### 2. Create venv with system site-packages
```bash
cd <output_dir>

if [ ! -d "venv" ]; then
  python3 -m venv venv --system-site-packages
  echo "Created venv with system site-packages access"
fi

source venv/bin/activate

python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import triton; print('Triton available')"
```

### 3. Install vLLM-rocm
```bash
source <output_dir>/venv/bin/activate
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || \
  pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

### 4. Install other missing packages
```bash
source <output_dir>/venv/bin/activate
python3 -c "import transformers" 2>/dev/null || pip install transformers
python3 -c "import accelerate" 2>/dev/null || pip install accelerate
```

### 5. Verify Installation
```bash
source <output_dir>/venv/bin/activate
python3 -c "
import torch, vllm
print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'vLLM {vllm.__version__}')
print(f'GPU: {torch.cuda.get_device_name()}')
"
```

### 6. Update progress.json
Update progress.json: phase="env", phases_completed.append("env")


---

# Phase 1: Model Serving with vLLM 

## Goal
Start the model using `vllm serve` and verify it works. vLLM handles model download automatically.

## ⚠️ vLLM Mode
In vLLM mode, there is NO need to:
- Manually download the model (vLLM auto-downloads from HuggingFace)
- Write a demo inference script
- Fix compatibility issues manually

## Steps

### 1. Test vLLM serve
```bash
source <output_dir>/venv/bin/activate

# Quick test: start vllm serve and send a test request
# Adjust --tensor-parallel-size based on model size and GPU count
vllm serve $1 \
  --dtype auto \
  --max-model-len 2048 \
  --port 8192 \
  --disable-log-requests &

VLLM_PID=$!
sleep 30  # Wait for model to load

# Test with a simple request
curl -s http://localhost:8192/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "$1", "prompt": "Hello, I am", "max_tokens": 20}' | python3 -m json.tool

# Kill the test server
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
```

### 2. Record model config
```bash
source <output_dir>/venv/bin/activate
python3 -c "
from transformers import AutoConfig
config = AutoConfig.from_pretrained('$1', trust_remote_code=True)
import json
info = {
    'model_type': getattr(config, 'model_type', 'unknown'),
    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
    'hidden_size': getattr(config, 'hidden_size', None),
    'num_attention_heads': getattr(config, 'num_attention_heads', None),
    'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
    'intermediate_size': getattr(config, 'intermediate_size', None),
    'vocab_size': getattr(config, 'vocab_size', None),
}
print(json.dumps(info, indent=2))
with open('<output_dir>/model_config.json', 'w') as f:
    json.dump(info, f, indent=2)
"
```

### 3. Update progress.json
Update progress.json: phases_completed.append("download"), phases_completed.append("demo"), phases_completed.append("compatibility")

> **Note**: In vLLM mode, Phase 1 covers download + demo + compatibility in one step.


---

# Phase 2: (Covered by Phase 1 in vLLM mode) 

> In vLLM mode, demo generation is handled by Phase 1 (`vllm serve`). Skip this phase.

Update progress.json if not already done.


---

# Phase 3: (Covered by Phase 1 in vLLM mode) 

> In vLLM mode, compatibility fixes are handled by vLLM itself. Skip this phase.

If vLLM serve failed in Phase 1, debug using vLLM logs (check `--dtype`, `--tensor-parallel-size`, `--max-model-len`).

Update progress.json if not already done.


---

# Phase 4: Performance Profiling 

## Goal
Benchmark vLLM serving throughput AND collect GPU kernel trace for bottleneck analysis.

## ⚠️ CORRECT Profiling Approach for vLLM

**DO NOT** use `rocprof` or single-request profiling.
**DO** use `vllm bench serve` with realistic concurrency to capture production-like behavior.

The profiling has TWO parts:
1. **Throughput benchmark**: Measure baseline ITPS/OTPS/TPOT/TTFT at target concurrency
2. **Kernel trace**: Collect torch profiler trace via `VLLM_TORCH_PROFILER_DIR` for kernel analysis

## Step 1: Baseline Throughput Benchmark

```bash
source <output_dir>/venv/bin/activate

# Start vLLM serve (baseline, no profiler)
vllm serve $1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8192 \
  --disable-log-requests &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM to be ready..."
timeout 300 bash -c 'until curl -s http://localhost:8192/health > /dev/null 2>&1; do sleep 5; done'
echo "Server ready!"

# Run benchmark: 1K input / 1K output, concurrency=16
vllm bench serve \
  --model $1 \
  --port 8192 \
  --dataset-name random \
  --input-len 1024 \
  --output-len 1024 \
  --num-prompts 100 \
  --max-concurrency 16 \
  --request-rate inf \
  --save-result \
  --result-dir <output_dir>/profile \
  --result-filename baseline_benchmark.json \
  --label baseline

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

Parse and save the key metrics:
```bash
python3 -c "
import json
with open('<output_dir>/profile/baseline_benchmark.json') as f:
    data = json.load(f)
print('=== Baseline Throughput ===')
for key in ['total_input_tokens', 'total_output_tokens', 'request_throughput',
            'input_throughput', 'output_throughput',
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
            'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
            'mean_itl_ms', 'median_itl_ms', 'p99_itl_ms']:
    val = data.get(key, 'N/A')
    print(f'  {key}: {val}')
"
```

## Step 2: Collect Kernel Trace

```bash
source <output_dir>/venv/bin/activate
mkdir -p <output_dir>/profile/traces

# Start vLLM WITH profiler enabled
VLLM_TORCH_PROFILER_DIR=<output_dir>/profile/traces \
vllm serve $1 \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --disable-log-requests &
VLLM_PID=$!

echo "Waiting for vLLM (profiler) to be ready..."
timeout 300 bash -c 'until curl -s http://localhost:8193/health > /dev/null 2>&1; do sleep 5; done'

# Send requests for trace collection (fewer prompts, same concurrency)
vllm bench serve \
  --model $1 \
  --port 8193 \
  --dataset-name random \
  --input-len 1024 \
  --output-len 1024 \
  --num-prompts 30 \
  --max-concurrency 16 \
  --request-rate inf \
  --save-result \
  --result-dir <output_dir>/profile \
  --result-filename trace_benchmark.json \
  --label trace

# Wait for profiler to flush
sleep 15

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

echo "Trace files:"
ls -lh <output_dir>/profile/traces/
```

## Step 3: Extract Kernel Bottlenecks from Trace

```bash
cd <output_dir>/profile
cp <output_dir>/scripts/vllm_trace_extractor.py .

# Find latest trace file
TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
echo "Analyzing: $TRACE_FILE"

python3 vllm_trace_extractor.py -i "$TRACE_FILE" \
  --full-csv kernel_full.csv \
  --unique-csv kernel_unique.csv
```

## Step 4: Generate bottlenecks.json

```bash
cd <output_dir>/profile
python3 -c "
import csv, json

kernels = []
with open('kernel_unique.csv') as f:
    for row in csv.DictReader(f):
        kernels.append({
            'name': row['name'], 'count': int(row['count']),
            'total_dur_us': float(row['total_dur']),
            'avg_dur_us': float(row['avg_dur']),
            'median_dur_us': float(row['median_dur']),
        })

total = sum(k['total_dur_us'] for k in kernels)
bottlenecks = []
for k in kernels[:30]:
    pct = k['total_dur_us'] / total * 100 if total > 0 else 0
    name = k['name']
    optimizable = True
    reason = ''
    if 'Cijk_' in name or 'gemm' in name.lower() or 'hipblas' in name.lower():
        reason = 'GEMM/rocBLAS'; optimizable = False
    elif 'attn' in name.lower() or 'flash' in name.lower() or 'mha' in name.lower():
        reason = 'Attention'; optimizable = True
    elif 'norm' in name.lower() or 'rms' in name.lower():
        reason = 'Normalization - Triton fusable'; optimizable = True
    elif 'elementwise' in name.lower() or 'vectorized' in name.lower():
        reason = 'Elementwise - Triton fusable'; optimizable = True
    elif 'silu' in name.lower() or 'gelu' in name.lower() or 'act' in name.lower():
        reason = 'Activation - fusable'; optimizable = True
    elif 'rope' in name.lower() or 'rotary' in name.lower():
        reason = 'RoPE - Triton fusable'; optimizable = True
    elif 'copy' in name.lower() or 'Cat' in name:
        reason = 'Memory op'; optimizable = False
    bottlenecks.append({**k, 'cuda_time_percent': pct, 'optimizable': optimizable, 'reason': reason})

print(f'Total GPU time: {total/1000:.2f}ms')
for i, b in enumerate(bottlenecks[:15], 1):
    opt = '✓' if b['optimizable'] else '✗'
    print(f\"{i:2d}. {b['name'][:50]:50s} {b['cuda_time_percent']:5.1f}% ({b['total_dur_us']/1000:.2f}ms) x{b['count']} {opt} {b['reason']}\")

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
"
```

## Step 5: Save model shapes for problem file generation

```bash
source <output_dir>/venv/bin/activate
python3 -c "
import json
from transformers import AutoConfig
c = AutoConfig.from_pretrained('$1', trust_remote_code=True)
shapes = {
    'hidden_size': getattr(c, 'hidden_size', None),
    'intermediate_size': getattr(c, 'intermediate_size', None),
    'num_attention_heads': getattr(c, 'num_attention_heads', None),
    'num_key_value_heads': getattr(c, 'num_key_value_heads', None),
    'head_dim': getattr(c, 'hidden_size', 0) // max(getattr(c, 'num_attention_heads', 1), 1),
    'num_hidden_layers': getattr(c, 'num_hidden_layers', None),
    'vocab_size': getattr(c, 'vocab_size', None),
}
with open('<output_dir>/profile/model_shapes.json', 'w') as f:
    json.dump(shapes, f, indent=2)
print(json.dumps(shapes, indent=2))
"
```

Update progress.json: phases_completed.append("profile")


---

# Phase 5: Generate Problem Files for Kernel Optimization 

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 1: Operator Fusion Analysis (CRITICAL)

A standalone `analyze_fusion.py` script is provided at `<output_dir>/scripts/analyze_fusion.py`.
Use it to detect fusable operator patterns:

```bash
cp <output_dir>/scripts/analyze_fusion.py <output_dir>/profile/
cd <output_dir>/profile
python analyze_fusion.py
cat fusion_opportunities.json
```

### Common Fusion Opportunities in LLMs

| Pattern | Operators to Fuse | Fused Name | Expected Speedup |
|---------|-------------------|------------|------------------|
| **ResidualNorm** | add + rmsnorm/layernorm | fused_residual_norm | 1.2-1.5x |
| **SwiGLU/GeGLU** | silu/gelu + mul | fused_swiglu | 1.3-1.8x |
| **BiasAdd** | matmul + add (bias) | fused_linear_bias | 1.1-1.3x |
| **RotaryEmbed** | rope_cos + rope_sin + cat | fused_rope | 1.2-1.5x |
| **QKV Projection** | 3x linear (q,k,v) | fused_qkv_proj | 1.2-1.4x |
| **MLP Block** | linear + activation + linear | fused_mlp | 1.3-2.0x |

## STEP 2: Create FUSED Problem Files (Priority)

**Create fused kernels BEFORE individual kernels!**

Use ACTUAL shapes from `<output_dir>/profile/shape_ranges.json` or `<output_dir>/profile/bottlenecks.json`.

### Example: Fused Residual + RMSNorm
```python
# problem_fused_residual_rmsnorm.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = eps
    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

# ⚠️ Use shapes from shape_ranges.json!
batch_size = 1
seq_len = 64       # typical from profiling
hidden_size = 4096 # from model config

def get_inputs():
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
    ]
def get_init_inputs():
    return [hidden_size]
```

### Example: Fused SwiGLU
```python
# problem_fused_swiglu.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up

batch_size, seq_len, intermediate_size = 1, 64, 11008
def get_inputs():
    return [
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
    ]
def get_init_inputs():
    return []
```

## STEP 3: Create Individual Problem Files (Lower Priority)

Only for operators that: cannot be fused, take > 5% time, and are not already optimized by vendor libs (rocBLAS GEMM).

### Common Operators

- **RMSNorm/LayerNorm**: `class Model` with weight param, forward does variance + rsqrt + mul
- **Attention**: `class Model` wrapping `torch.nn.functional.scaled_dot_product_attention`
- **SiLU/GELU**: `class Model` with activation function
- **RoPE**: `class Model` with cos/sin rotation

## STEP 4: Generate Optimization Manifest

Create `<output_dir>/problems/optimization_manifest.json`:

```json
{
  "model": "$1",
  "description": "Edit 'enabled' to control which optimizations to apply",
  "optimizations": [
    {"name": "fused_residual_rmsnorm", "file": "problem_fused_residual_rmsnorm.py",
     "type": "fused", "priority": "HIGH", "enabled": true},
    {"name": "linear_gemm", "file": "problem_linear.py",
     "type": "individual", "priority": "LOW", "enabled": false,
     "notes": "rocBLAS usually optimal"}
  ]
}
```

## Steps
1. Run fusion analysis
2. Create fused problem files (HIGH priority)
3. Create individual problem files (MEDIUM/LOW)
4. Generate optimization_manifest.json
5. Update progress.json



---

# Phase 6: Kernel Optimization 

## Goal
Write optimized Triton kernels for each problem file and verify speedup.

## ⚠️ NO external `opencode` command needed
Optimize kernels DIRECTLY in this session using the test scripts provided.

## Scripts Available
- `<output_dir>/scripts/kernel_test_runner.py` — test accuracy + benchmark
- `<output_dir>/scripts/kernel_finalize.py` — save best result to target file

## Workflow for EACH Problem File

For each `problem_*.py` file in `<output_dir>/problems/`:

### 1. Read the source file to understand the PyTorch operator
```bash
cat <output_dir>/problems/problem_XXX.py
```

### 2. Check GPU architecture
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, Arch: {torch.cuda.get_device_capability()}')"
```

### 3. Write the optimized Triton kernel
Create `<output_dir>/problems/problem_XXX_opt.py` with:
- `class ModelNew(nn.Module)` using `@triton.jit` Triton kernels
- Same `__init__` signature as `Model`
- Use `@triton.autotune` with 10-20 diverse configs

### 4. Test accuracy + benchmark
```bash
source <output_dir>/venv/bin/activate
python3 <output_dir>/scripts/kernel_test_runner.py \
  --src <output_dir>/problems/problem_XXX.py \
  --target <output_dir>/problems/problem_XXX_opt.py
```

The script prints: `RESULT_JSON: {"speedup": 1.5, "accuracy": "PASSED", ...}`

### 5. Iterate if needed
- Accuracy FAILED → fix kernel, re-run step 4
- Speedup too low → adjust block sizes, fusion strategy, re-run step 4

### 6. Finalize when satisfied
```bash
python3 <output_dir>/scripts/kernel_finalize.py \
  --target <output_dir>/problems/problem_XXX_opt.py
```

## Priority Order

| Priority | Kernel Type | Goal | Reason |
|----------|-------------|------|--------|
| **HIGH** | Fused Residual+RMSNorm | 1.5x | Memory traffic reduction |
| **HIGH** | Fused SwiGLU | 1.5x | Activation fusion |
| **HIGH** | Fused RoPE | 1.5x | Custom optimization |
| MEDIUM | Individual norms | 1.3x | If not covered by fused version |
| LOW | Linear/GEMM | 1.1x | rocBLAS usually optimal |
| **SKIP** | Simple add/copy | — | Overhead > benefit |

## When to SKIP a kernel
- If it's part of a fused kernel you already optimized
- If rocBLAS/vendor lib is already near-optimal
- If after 3 attempts speedup is < 1.0x at actual shapes

## Triton Optimization Guide

### Autotune Strategy
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        # ... add 10-20 configs covering the space
    ],
    key=['N'],
)
```

### Common Patterns
- **Memory-bound**: Optimize access patterns, vectorization
- **Compute-bound**: Larger tiles, more arithmetic per memory access
- **Fused kernels**: Combine multiple ops to reduce memory traffic
- **FP32 accumulation**: Use `tl.float32` for acc, cast output at end

## After All Kernels Done

```bash
echo "=== Kernel Optimization Results ==="
cd <output_dir>/problems
for f in *_opt.py; do
  if [ -f "$f" ]; then
    tracker="${f%.py}_best.json"
    if [ -f "$tracker" ]; then
      speedup=$(python3 -c "import json; print(json.load(open('$tracker')).get('best_speedup', 0))")
      echo "  $f: ${speedup}x"
    fi
  fi
done
```

Copy successful optimizations to `<output_dir>/optimized/`:
```bash
cd <output_dir>/problems
for f in *_opt.py; do
  tracker="${f%.py}_best.json"
  if [ -f "$tracker" ]; then
    speedup=$(python3 -c "import json; d=json.load(open('$tracker')); print(d.get('best_speedup',0))")
    if python3 -c "exit(0 if $speedup > 1.0 else 1)"; then
      cp "$f" <output_dir>/optimized/
      echo "Copied $f (${speedup}x)"
    fi
  fi
done
```

Update progress.json: phases_completed.append("optimize")


---

# Phase 7: Integration & End-to-End Testing 

## Goal
Apply optimized kernels to vLLM via CustomOp and measure ACTUAL serving throughput.

## ⛔ MANDATORY: This phase REQUIRES real measured data

**This phase is NOT complete until:**
1. A patched vLLM server has ACTUALLY been started and served requests
2. `vllm bench serve` has been run against the patched server
3. `optimized_serving.json` has `"label": "optimized"` (NOT "baseline")
4. The validation script passes

**FORBIDDEN:**
- Estimating speedup with Amdahl's law
- Copying baseline numbers and modifying them
- Reporting "estimated" or "conservative" speedup
- Skipping the patched server benchmark

---

## Integration Mechanism: vLLM CustomOp.register_oot()

We use vLLM's OFFICIAL extension mechanism (not monkey-patching):
- Docs: https://docs.vllm.ai/en/latest/design/custom_op/
- Each optimized kernel is wrapped as a vLLM CustomOp subclass
- `CustomOp.register_oot()` replaces the default op at instantiation time
- If the optimized kernel fails, vLLM falls back to the default

---

## Step 1: Generate vLLM Plugin

The `generate_vllm_plugin.py` script auto-creates a plugin from `*_opt.py` files:

```bash
source <output_dir>/venv/bin/activate
cd <output_dir>/optimized

# Copy all *_opt.py from problems
cp <output_dir>/problems/*_opt.py . 2>/dev/null

# Generate the plugin
python3 <output_dir>/scripts/generate_vllm_plugin.py \
  --kernel-dir <output_dir>/optimized

# Verify generated files
ls -la vllm_plugin/
cat vllm_plugin/manifest.json
```

This generates:
- `<output_dir>/optimized/vllm_plugin/__init__.py` — registers CustomOps
- `<output_dir>/optimized/run_patched_vllm.py` — launcher script
- `<output_dir>/optimized/vllm_plugin/manifest.json` — registration summary

## Step 2: Test Plugin Registration (dry run)

Verify that the plugin loads without errors:

```bash
source <output_dir>/venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, '<output_dir>/optimized')
import vllm_plugin
print('Plugin loaded successfully')
"
```

## Step 3: ⛔ MANDATORY — Benchmark Baseline

Use existing `baseline_serving.json` from Phase 4, or re-run:

```bash
source <output_dir>/venv/bin/activate

vllm serve $1 --dtype auto --max-model-len 4096 --port 8192 --disable-log-requests &
VLLM_PID=$!
timeout 300 bash -c 'until curl -s http://localhost:8192/health >/dev/null 2>&1; do sleep 5; done'

vllm bench serve \
  --model $1 --port 8192 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/report --result-filename baseline_serving.json --label baseline

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

## Step 4: ⛔ MANDATORY — Start Patched vLLM and Benchmark

```bash
source <output_dir>/venv/bin/activate

# Start vLLM with CustomOp plugin
python3 <output_dir>/optimized/run_patched_vllm.py serve \
  --model $1 --dtype auto --max-model-len 4096 \
  --port 8193 --disable-log-requests &
PATCHED_PID=$!

# Wait for server to be ready
echo "Waiting for patched vLLM..."
timeout 300 bash -c 'until curl -s http://localhost:8193/health >/dev/null 2>&1; do sleep 5; done'
echo "Patched server ready!"

# Verify the correct model is loaded
curl -s http://localhost:8193/v1/models | python3 -c "
import json,sys
data=json.load(sys.stdin)
models=[m['id'] for m in data.get('data',[])]
print(f'Models: {models}')
assert '$1' in models, f'Expected $1 but got {models}'
print('✓ Correct model loaded')
"

# Quick correctness test — verify server responds
curl -s http://localhost:8193/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"$1","prompt":"Hello","max_tokens":5}' | python3 -m json.tool

# Run benchmark (SAME parameters as baseline)
vllm bench serve \
  --model $1 --port 8193 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/report --result-filename optimized_serving.json --label optimized

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null
```

**If the patched server fails to start or crashes:**
1. Check `run_patched_vllm.py` output for registration errors
2. Try removing problematic kernels from `vllm_plugin/` and regenerate
3. If ALL patches fail, run benchmark anyway (it measures "no-change" as the honest result)

## Step 5: ⛔ MANDATORY — Validate Results

```bash
source <output_dir>/venv/bin/activate
python3 << 'VALIDATE'
import json, sys, os

report_dir = "<output_dir>/report"
baseline_path = os.path.join(report_dir, "baseline_serving.json")
optimized_path = os.path.join(report_dir, "optimized_serving.json")
errors = []

for path, name, expected_label in [
    (baseline_path, "baseline", "baseline"),
    (optimized_path, "optimized", "optimized"),
]:
    if not os.path.exists(path):
        errors.append(f"MISSING: {name}_serving.json — you must run vllm bench serve")
        continue
    with open(path) as f:
        data = json.load(f)
    label = data.get("label", "")
    if label != expected_label:
        errors.append(f"{name}_serving.json label='{label}', expected '{expected_label}'")
    completed = data.get("completed", 0)
    if completed == 0 and name == "optimized":
        errors.append(f"optimized_serving.json has completed=0 — patched server did not work")
    if name == "optimized" and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        if data.get("date") == bl.get("date"):
            errors.append("SUSPICIOUS: same date on baseline and optimized — were these separate runs?")

if errors:
    print("⛔ VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    print("\nYou must fix the issues above. Phase 7 is NOT complete.")
    sys.exit(1)

with open(baseline_path) as f: baseline = json.load(f)
with open(optimized_path) as f: optimized = json.load(f)

b_otps = baseline.get("output_throughput", 0)
o_otps = optimized.get("output_throughput", 0)
speedup = o_otps / b_otps if b_otps > 0 else 1.0

print("✅ VALIDATION PASSED — Real measurements confirmed")
print(f"  Baseline OTPS:  {b_otps:.2f} tok/s (completed={baseline.get('completed',0)})")
print(f"  Optimized OTPS: {o_otps:.2f} tok/s (completed={optimized.get('completed',0)})")
print(f"  Speedup:        {speedup:.3f}x")

os.makedirs(os.path.join(report_dir, "comparison_outputs"), exist_ok=True)
with open(os.path.join(report_dir, "comparison_outputs", "comparison_results.json"), "w") as f:
    json.dump({
        "validated": True,
        "baseline_otps": b_otps, "optimized_otps": o_otps, "speedup_otps": speedup,
        "baseline_tpot_ms": baseline.get("mean_tpot_ms", 0),
        "optimized_tpot_ms": optimized.get("mean_tpot_ms", 0),
        "baseline_ttft_ms": baseline.get("mean_ttft_ms", 0),
        "optimized_ttft_ms": optimized.get("mean_ttft_ms", 0),
        "concurrency": 16, "input_len": 1024, "output_len": 1024,
    }, f, indent=2)
VALIDATE
```

**If validation fails, fix the issue and re-run from the failing step.**

Update progress.json: phases_completed.append("integrate")


---

# Phase 8: Generate Final Report 

## Goal
Create a comprehensive optimization report.

## Create Report: `<output_dir>/report/optimization_report.md`

**⚠️ CRITICAL**: Include **ACTUAL MEASURED** end-to-end speedup from comparison_results.json.

```markdown
# Model Optimization Report

## Model Information
- **Model**: $1
- **Optimization Date**: [DATE]

## Summary
- **ACTUAL End-to-End Speedup**: X.Xx (measured, NOT estimated)
- **Kernels Optimized**: N
- **Baseline Inference Time**: X.Xs
- **Optimized Inference Time**: X.Xs

## Bottleneck Analysis
| Operator | Original Time (ms) | % of Total | Optimized | Speedup |
|----------|-------------------|------------|-----------|---------|
| ...      | ...               | ...        | ...       | ...     |

## Performance Results (ACTUAL MEASURED)
| Metric | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| End-to-End Inference Time | X.Xs | X.Xs | **X.Xx** |

## Comparison Outputs (Seed=42)
Outputs generated with fixed random seed for verification.

### Text Models:
| Original | Optimized |
|----------|-----------|
| [text]   | [text]    |

### Image Models:
| Original | Optimized |
|:--------:|:---------:|
| ![Original](comparison_outputs/original_output.png) | ![Optimized](comparison_outputs/optimized_output.png) |

## Files Generated
- model/ - Downloaded model
- demo/demo.py - Working demo
- profile/bottlenecks.json - Profiling results
- problems/ - Problem files + optimized kernels
- optimized/integrate.py - Integration script
- report/optimization_report.md - This report

## Recommendations
1. ...
```

## Steps
1. Gather all results from previous phases
2. Generate the comprehensive report
3. Update progress.json: phase="complete", phases_completed.append("report")



---

# EXECUTION INSTRUCTIONS

## Execute phases in order: 0 → 1 → 4 → 5 → 6 → 7 → 8
(Phases 2 and 3 are handled by vLLM automatically)

## General Rules
1. **Update progress.json after each phase**
2. **If a phase fails, debug and fix before proceeding**
3. **Use kernel_test_runner.py for kernel testing** (no external `opencode` command needed)
4. **NEVER modify system libraries — only use project venv**
5. **Phase 7 MUST have real measured data — no estimates**

Begin with Phase 0: Environment Setup.
