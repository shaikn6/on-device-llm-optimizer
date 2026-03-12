#!/usr/bin/env bash
# quantize-and-deploy.sh
# Download a HuggingFace model → quantize to INT4 (GGUF) → benchmark → push to registry
#
# Usage:
#   ./scripts/quantize-and-deploy.sh --model mistralai/Mistral-7B-v0.1 --bits 4 --push
#   ./scripts/quantize-and-deploy.sh --model meta-llama/Llama-3-8B --bits 8
#   ./scripts/quantize-and-deploy.sh --model Qwen/Qwen2-7B --bits 4 --quantization-type q4_k_m
#
# Dependencies:
#   - Python >= 3.10 with transformers, torch, optimum installed (see requirements.txt)
#   - llama.cpp compiled: brew install llama.cpp  OR  build from source
#   - jq, bc for benchmark parsing
#   - HF_TOKEN env var set (for gated models)
#
# Output:
#   - ./checkpoints/<model_name>_<bits>bit.gguf
#   - ./results/benchmark_<model_name>_<bits>bit.json

set -euo pipefail
IFS=$'\n\t'

# ── Script location ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────────────────
MODEL_ID=""
BITS=4
QUANT_TYPE="q4_k_m"        # llama.cpp quantization type
CONTEXT_SIZE=4096
BENCHMARK_TOKENS=512
N_BENCH_RUNS=3
OUTPUT_DIR="${REPO_ROOT}/checkpoints"
RESULTS_DIR="${REPO_ROOT}/results"
HF_CACHE_DIR="${HOME}/.cache/huggingface/hub"
PUSH_TO_REGISTRY=false
REGISTRY_IMAGE_BASE="ghcr.io/shaikn6/on-device-llm"
DEVICE="cpu"               # cpu | cuda | mps
SKIP_DOWNLOAD=false
SKIP_BENCHMARK=false

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_success() { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --model         HuggingFace model ID (required) e.g. mistralai/Mistral-7B-v0.1
  --bits          Quantization bits: 4 or 8 (default: 4)
  --quant-type    llama.cpp quant type (default: q4_k_m)
                  Options: q4_0, q4_1, q4_k_m, q4_k_s, q5_0, q5_1, q5_k_m,
                           q6_k, q8_0, f16
  --context       Context window size (default: 4096)
  --device        Inference device: cpu|cuda|mps (default: cpu)
  --push          Push GGUF + benchmark to container registry
  --skip-download Use existing HF cache (skip re-download)
  --skip-benchmark Skip llama-bench run
  --output-dir    Output dir for .gguf files (default: ./checkpoints)
  --results-dir   Output dir for benchmark JSON (default: ./results)
  -h, --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)          MODEL_ID="$2";       shift 2 ;;
    --bits)           BITS="$2";           shift 2 ;;
    --quant-type)     QUANT_TYPE="$2";     shift 2 ;;
    --context)        CONTEXT_SIZE="$2";   shift 2 ;;
    --device)         DEVICE="$2";         shift 2 ;;
    --push)           PUSH_TO_REGISTRY=true; shift ;;
    --skip-download)  SKIP_DOWNLOAD=true;  shift ;;
    --skip-benchmark) SKIP_BENCHMARK=true; shift ;;
    --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
    --results-dir)    RESULTS_DIR="$2";    shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *)                log_error "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "${MODEL_ID}" ]]; then
  log_error "Missing required argument: --model"
  usage; exit 1
fi

if [[ "${BITS}" != "4" && "${BITS}" != "8" ]]; then
  log_error "--bits must be 4 or 8 (got: ${BITS})"
  exit 1
fi

# ── Tool checks ───────────────────────────────────────────────────────────────
for tool in python3 jq bc; do
  if ! command -v "${tool}" &>/dev/null; then
    log_error "Required tool not found: ${tool}"
    exit 1
  fi
done

# Find llama.cpp tools
LLAMA_CONVERT=""
LLAMA_QUANTIZE=""
LLAMA_BENCH=""

for candidate in convert_hf_to_gguf.py convert.py; do
  if python3 -c "import llama_cpp" &>/dev/null; then
    LLAMA_CONVERT="python3 -m llama_cpp.convert_hf_to_gguf"
    break
  fi
  # Check if llama.cpp is compiled locally
  if [[ -f "${REPO_ROOT}/llama.cpp/convert_hf_to_gguf.py" ]]; then
    LLAMA_CONVERT="python3 ${REPO_ROOT}/llama.cpp/convert_hf_to_gguf.py"
    break
  fi
done

if command -v llama-quantize &>/dev/null; then
  LLAMA_QUANTIZE="llama-quantize"
elif [[ -f "${REPO_ROOT}/llama.cpp/llama-quantize" ]]; then
  LLAMA_QUANTIZE="${REPO_ROOT}/llama.cpp/llama-quantize"
else
  log_error "llama-quantize not found. Install with: brew install llama.cpp"
  exit 1
fi

if command -v llama-bench &>/dev/null; then
  LLAMA_BENCH="llama-bench"
elif [[ -f "${REPO_ROOT}/llama.cpp/llama-bench" ]]; then
  LLAMA_BENCH="${REPO_ROOT}/llama.cpp/llama-bench"
fi

# ── Setup directories ─────────────────────────────────────────────────────────
MODEL_SLUG="${MODEL_ID//\//_}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

HF_MODEL_DIR="${HF_CACHE_DIR}/models--${MODEL_ID//\//_}"
FP16_GGUF="${OUTPUT_DIR}/${MODEL_SLUG}_fp16.gguf"
QUANT_GGUF="${OUTPUT_DIR}/${MODEL_SLUG}_${BITS}bit_${QUANT_TYPE}.gguf"
BENCHMARK_JSON="${RESULTS_DIR}/benchmark_${MODEL_SLUG}_${BITS}bit_${TIMESTAMP}.json"
BENCHMARK_LATEST="${RESULTS_DIR}/benchmark_${MODEL_SLUG}_latest.json"

log_info "======================================================="
log_info " on-device-llm-optimizer: quantize-and-deploy"
log_info "======================================================="
log_info " Model:      ${MODEL_ID}"
log_info " Bits:       ${BITS}"
log_info " Quant type: ${QUANT_TYPE}"
log_info " Context:    ${CONTEXT_SIZE}"
log_info " Device:     ${DEVICE}"
log_info " Output:     ${QUANT_GGUF}"
log_info "======================================================="

# ── Step 1: Download model from HuggingFace ───────────────────────────────────
if [[ "${SKIP_DOWNLOAD}" == "true" ]]; then
  log_info "Skipping download (--skip-download set)"
else
  log_info "Step 1/4: Downloading ${MODEL_ID} from HuggingFace Hub..."

  DOWNLOAD_START=$(date +%s)
  python3 - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download

model_id = "${MODEL_ID}"
cache_dir = os.environ.get("HF_HOME", "${HF_CACHE_DIR}")
token = os.environ.get("HF_TOKEN")

print(f"Downloading {model_id} to {cache_dir}...")
local_dir = snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    token=token,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
)
print(f"Downloaded to: {local_dir}")
# Write the local path to a temp file so bash can read it
with open("/tmp/hf_model_local_dir", "w") as f:
    f.write(local_dir)
PYEOF

  DOWNLOAD_END=$(date +%s)
  DOWNLOAD_SECS=$((DOWNLOAD_END - DOWNLOAD_START))
  log_success "Download complete in ${DOWNLOAD_SECS}s"
fi

HF_LOCAL_DIR="$(cat /tmp/hf_model_local_dir 2>/dev/null || echo "${HF_MODEL_DIR}")"

# ── Step 2: Convert FP32/BF16 model to FP16 GGUF ─────────────────────────────
log_info "Step 2/4: Converting ${MODEL_ID} to FP16 GGUF..."
CONVERT_START=$(date +%s)

if [[ -n "${LLAMA_CONVERT}" ]]; then
  ${LLAMA_CONVERT} \
    "${HF_LOCAL_DIR}" \
    --outtype f16 \
    --outfile "${FP16_GGUF}"
else
  # Fallback: use optimum-cli if llama.cpp convert not found
  log_warn "llama.cpp convert not found; trying optimum-cli..."
  if command -v optimum-cli &>/dev/null; then
    optimum-cli export gguf \
      --model "${MODEL_ID}" \
      --output "${OUTPUT_DIR}" \
      --dtype float16
    # Rename to expected path
    mv "${OUTPUT_DIR}"/*.gguf "${FP16_GGUF}" 2>/dev/null || true
  else
    log_error "No GGUF conversion tool found. Install llama.cpp or optimum[gguf]."
    exit 1
  fi
fi

CONVERT_END=$(date +%s)
CONVERT_SECS=$((CONVERT_END - CONVERT_START))
FP16_SIZE=$(du -sh "${FP16_GGUF}" | awk '{print $1}')
log_success "FP16 GGUF: ${FP16_GGUF} (${FP16_SIZE}) in ${CONVERT_SECS}s"

# ── Step 3: Quantize FP16 GGUF → INT4 GGUF ───────────────────────────────────
log_info "Step 3/4: Quantizing to ${BITS}-bit (${QUANT_TYPE})..."
QUANT_START=$(date +%s)

${LLAMA_QUANTIZE} \
  "${FP16_GGUF}" \
  "${QUANT_GGUF}" \
  "${QUANT_TYPE}" \
  2>&1 | tee /tmp/quantize_log.txt

QUANT_END=$(date +%s)
QUANT_SECS=$((QUANT_END - QUANT_START))
QUANT_SIZE=$(du -sh "${QUANT_GGUF}" | awk '{print $1}')
QUANT_SIZE_BYTES=$(stat -f%z "${QUANT_GGUF}" 2>/dev/null || stat -c%s "${QUANT_GGUF}")
log_success "Quantized GGUF: ${QUANT_GGUF} (${QUANT_SIZE}) in ${QUANT_SECS}s"

# Compute compression ratio
FP16_SIZE_BYTES=$(stat -f%z "${FP16_GGUF}" 2>/dev/null || stat -c%s "${FP16_GGUF}")
COMPRESSION_RATIO=$(echo "scale=2; ${FP16_SIZE_BYTES} / ${QUANT_SIZE_BYTES}" | bc)
log_info "Compression ratio vs FP16: ${COMPRESSION_RATIO}x"

# ── Step 4: Benchmark with llama-bench ────────────────────────────────────────
if [[ "${SKIP_BENCHMARK}" == "true" ]]; then
  log_info "Skipping benchmark (--skip-benchmark set)"
  PP_TPS="N/A"
  TG_TPS="N/A"
else
  log_info "Step 4/4: Benchmarking ${QUANT_GGUF}..."

  if [[ -z "${LLAMA_BENCH}" ]]; then
    log_warn "llama-bench not found — skipping benchmark"
    PP_TPS="N/A"
    TG_TPS="N/A"
  else
    BENCH_START=$(date +%s)

    ${LLAMA_BENCH} \
      --model "${QUANT_GGUF}" \
      --ctx-size "${CONTEXT_SIZE}" \
      --n-prompt 512 \
      --n-gen "${BENCHMARK_TOKENS}" \
      --n-thread "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)" \
      --output json \
      > /tmp/bench_raw.json 2>&1

    BENCH_END=$(date +%s)
    BENCH_SECS=$((BENCH_END - BENCH_START))

    PP_TPS=$(jq -r '.[0].pp_avg // "N/A"' /tmp/bench_raw.json 2>/dev/null || echo "N/A")
    TG_TPS=$(jq -r '.[0].tg_avg // "N/A"' /tmp/bench_raw.json 2>/dev/null || echo "N/A")

    log_success "Prompt processing: ${PP_TPS} tok/s"
    log_success "Token generation:  ${TG_TPS} tok/s"
    log_info "Benchmark complete in ${BENCH_SECS}s"
  fi
fi

# ── Write benchmark report ────────────────────────────────────────────────────
REPORT_JSON=$(cat <<JSONEOF
{
  "timestamp": "${TIMESTAMP}",
  "model_id": "${MODEL_ID}",
  "model_slug": "${MODEL_SLUG}",
  "quantization": {
    "bits": ${BITS},
    "type": "${QUANT_TYPE}",
    "method": "llama.cpp"
  },
  "files": {
    "fp16_gguf": "${FP16_GGUF}",
    "quant_gguf": "${QUANT_GGUF}",
    "fp16_size_bytes": ${FP16_SIZE_BYTES},
    "quant_size_bytes": ${QUANT_SIZE_BYTES},
    "compression_ratio": ${COMPRESSION_RATIO}
  },
  "benchmark": {
    "device": "${DEVICE}",
    "context_size": ${CONTEXT_SIZE},
    "benchmark_tokens": ${BENCHMARK_TOKENS},
    "prompt_processing_tps": "${PP_TPS}",
    "token_generation_tps": "${TG_TPS}"
  },
  "timings_seconds": {
    "download": ${DOWNLOAD_SECS:-0},
    "convert_to_fp16": ${CONVERT_SECS:-0},
    "quantize": ${QUANT_SECS:-0}
  }
}
JSONEOF
)

echo "${REPORT_JSON}" > "${BENCHMARK_JSON}"
cp "${BENCHMARK_JSON}" "${BENCHMARK_LATEST}"
log_success "Benchmark report: ${BENCHMARK_JSON}"
log_info ""
jq . "${BENCHMARK_JSON}"

# ── Optional: push GGUF to container registry via OCI artifact ────────────────
if [[ "${PUSH_TO_REGISTRY}" == "true" ]]; then
  log_info ""
  log_info "Pushing GGUF to OCI registry as artifact..."

  if ! command -v oras &>/dev/null; then
    log_error "oras CLI not found. Install: https://oras.land/docs/installation"
    exit 1
  fi

  REGISTRY_TAG="${REGISTRY_IMAGE_BASE}:${MODEL_SLUG}-${BITS}bit-${QUANT_TYPE}"

  oras push "${REGISTRY_TAG}" \
    --artifact-type "application/vnd.gguf.model" \
    "${QUANT_GGUF}:application/vnd.gguf.model.v1" \
    "${BENCHMARK_JSON}:application/json"

  log_success "Pushed: ${REGISTRY_TAG}"
fi

log_info ""
log_success "======================================================"
log_success " quantize-and-deploy complete"
log_success "======================================================"
log_success " Input:       ${MODEL_ID}"
log_success " Output:      ${QUANT_GGUF} (${QUANT_SIZE})"
log_success " Compression: ${COMPRESSION_RATIO}x vs FP16"
log_success " PP speed:    ${PP_TPS} tok/s"
log_success " TG speed:    ${TG_TPS} tok/s"
log_success "======================================================"
