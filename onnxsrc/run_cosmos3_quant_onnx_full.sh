#!/usr/bin/env bash
set -euo pipefail

# =========================
# 用户配置区
# 直接修改这里即可；仍可用同名环境变量临时覆盖。
# =========================
COSMOS3_REPO_DIR="${COSMOS3_REPO_DIR:-/srv/data2/c00932551/Nvidia_models/cosmos-framework/cosmos_framework}"
COSMOS3_ONNX_SRC_DIR="${COSMOS3_REPO_DIR}/onnxsrc"
COSMOS3_CHECKPOINT_PATH="${COSMOS3_CHECKPOINT_PATH:-/srv/data2/c00932551/Nvidia_models/Cosmos3-Edge-Policy-DROID}"
COSMOS3_DOPT_SIM_PATH="${COSMOS3_DOPT_SIM_PATH:-/srv/data2/c00932551/Nvidia_models/cosmos_onnx_scripts/npu_hardware_sim}"
COSMOS3_LAYOUT_MANIFEST="${COSMOS3_LAYOUT_MANIFEST:-/srv/data2/c00932551/Nvidia_models/cosmos_policy_onnx/edge_policy.fp32.onnx.json}"
COSMOS3_CONDITION_VISION_FRAMES="${COSMOS3_CONDITION_VISION_FRAMES:-0}"
COSMOS3_CONDITION_ACTION_FRAMES="${COSMOS3_CONDITION_ACTION_FRAMES:-0}"
COSMOS3_EMBEDDING_SEPARATE="${COSMOS3_EMBEDDING_SEPARATE:-0}"  # 默认只生成主量化文件
COSMOS3_ONNX_EXPORTER="${COSMOS3_ONNX_EXPORTER:-dynamo}"  # 保持原 Policy rewrite 的图契约
COSMOS3_EXTERNAL_DATA_MODE="${COSMOS3_EXTERNAL_DATA_MODE:-single}"  # 合并为单个 .onnx.data
COSMOS3_CLEAN_OUTPUT="${COSMOS3_CLEAN_OUTPUT:-1}"  # 默认删除旧输出，避免历史文件混入

if [[ ! -d "${COSMOS3_REPO_DIR}" ]]; then
    echo "Cosmos3 repository directory not found: ${COSMOS3_REPO_DIR}" >&2
    exit 1
fi
if [[ ! -e "${COSMOS3_CHECKPOINT_PATH}" ]]; then
    echo "Cosmos3 checkpoint not found: ${COSMOS3_CHECKPOINT_PATH}" >&2
    echo "Edit COSMOS3_CHECKPOINT_PATH in this script." >&2
    exit 1
fi
if [[ ! -d "${COSMOS3_DOPT_SIM_PATH}" ]]; then
    echo "DOPT simulator directory not found: ${COSMOS3_DOPT_SIM_PATH}" >&2
    exit 1
fi
if [[ ! -f "${COSMOS3_LAYOUT_MANIFEST}" ]]; then
    echo "FP32 ONNX layout manifest not found: ${COSMOS3_LAYOUT_MANIFEST}" >&2
    echo "Edit COSMOS3_LAYOUT_MANIFEST in this script." >&2
    exit 1
fi

COSMOS3_IMAGE_HEIGHT="${COSMOS3_IMAGE_HEIGHT:-540}"
COSMOS3_IMAGE_WIDTH="${COSMOS3_IMAGE_WIDTH:-640}"
COSMOS3_ACTION_CHUNK_SIZE="${COSMOS3_ACTION_CHUNK_SIZE:-16}"
COSMOS3_ACTION_DIM="${COSMOS3_ACTION_DIM:-8}"
COSMOS3_CONDITIONING_FPS="${COSMOS3_CONDITIONING_FPS:-5}"
COSMOS3_RESOLUTION="${COSMOS3_RESOLUTION:-480}"
COSMOS3_DOMAIN_NAME="${COSMOS3_DOMAIN_NAME:-droid_lerobot}"
COSMOS3_PROMPT="${COSMOS3_PROMPT:-Pick up the object and place it in the target location.}"
COSMOS3_QUANT_BIT=8

COSMOS3_RESULT_ROOT="${COSMOS3_RESULT_ROOT:-/srv/data2/c00932551/Nvidia_models/cosmos_policy_onnx/quant}"
COSMOS3_QUANT_ROOT="${COSMOS3_QUANT_ROOT:-${COSMOS3_RESULT_ROOT}/edge_policy_${COSMOS3_ACTION_CHUNK_SIZE}actions_int8_dyn_s8}"

if [[ "${COSMOS3_CLEAN_OUTPUT}" == "1" ]]; then
    case "${COSMOS3_QUANT_ROOT}" in
        "${COSMOS3_RESULT_ROOT}"/*) rm -rf -- "${COSMOS3_QUANT_ROOT}" ;;
        *) echo "Refusing to clean path outside COSMOS3_RESULT_ROOT: ${COSMOS3_QUANT_ROOT}" >&2; exit 2 ;;
    esac
fi
mkdir -p "${COSMOS3_QUANT_ROOT}"

COSMOS3_QUANT_CONFIG="${COSMOS3_QUANT_ROOT}/dopt_config_int8_dyn_s8.json"
COSMOS3_QUANT_CONFIG_REPORT="${COSMOS3_QUANT_ROOT}/dopt_config_int8_dyn_s8.report.json"
COSMOS3_FAKEQUANT_WEIGHT="${COSMOS3_QUANT_ROOT}/fakequant_weight.pth"
COSMOS3_QUANT_OUTPUT_DIR="${COSMOS3_QUANT_ROOT}/quant_params"
COSMOS3_QUANT_PARAMS_FILE="${COSMOS3_QUANT_PARAMS_FILE:-${COSMOS3_QUANT_OUTPUT_DIR}_v2}"
COSMOS3_ONNX_RAW="${COSMOS3_QUANT_ROOT}/edge_policy.int8_fakequant.onnx"
COSMOS3_ONNX_COMPATIBLE="${COSMOS3_QUANT_ROOT}/edge_policy.int8_fakequant.compatible.onnx"
COSMOS3_ONNX_FINAL="${COSMOS3_QUANT_ROOT}/edge_policy.int8_fakequant.compatible.named.onnx"
COSMOS3_REWRITE_REPORT="${COSMOS3_ONNX_COMPATIBLE}.rewrite.json"
COSMOS3_PROMPT_EMBEDDING="${COSMOS3_ONNX_COMPATIBLE}.prompt_embedding.npy"
COSMOS3_FINALIZE_REPORT="${COSMOS3_QUANT_ROOT}/finalize_names.json"
COSMOS3_MATCH_REPORT="${COSMOS3_QUANT_ROOT}/quant_onnx_name_match.json"
COSMOS3_AUDIT_REPORT="${COSMOS3_QUANT_ROOT}/onnx_audit.json"

COMMON_ARGS=(
    --checkpoint-path "${COSMOS3_CHECKPOINT_PATH}"
    --dopt-sim-path "${COSMOS3_DOPT_SIM_PATH}"
    --quant-config "${COSMOS3_QUANT_CONFIG}"
    --fakequant-weight "${COSMOS3_FAKEQUANT_WEIGHT}"
    --layout-manifest "${COSMOS3_LAYOUT_MANIFEST}"
    --condition-vision-frames "${COSMOS3_CONDITION_VISION_FRAMES}"
    --condition-action-frames "${COSMOS3_CONDITION_ACTION_FRAMES}"
    --prompt "${COSMOS3_PROMPT}"
    --image-height "${COSMOS3_IMAGE_HEIGHT}"
    --image-width "${COSMOS3_IMAGE_WIDTH}"
    --action-chunk-size "${COSMOS3_ACTION_CHUNK_SIZE}"
    --action-dim "${COSMOS3_ACTION_DIM}"
    --conditioning-fps "${COSMOS3_CONDITIONING_FPS}"
    --resolution "${COSMOS3_RESOLUTION}"
    --domain-name "${COSMOS3_DOMAIN_NAME}"
)

cd "${COSMOS3_REPO_DIR}"

echo "========== Step 1: generate DOPT config =========="
python3 "${COSMOS3_ONNX_SRC_DIR}/cosmos3_quantize.py" \
    --stage gen-config \
    --quant-output-dir "${COSMOS3_QUANT_OUTPUT_DIR}" \
    --embedding-separate "${COSMOS3_EMBEDDING_SEPARATE}" \
    "${COMMON_ARGS[@]}"

echo "========== Step 2: select INT8 dyn_s8 Linear layers =========="
python3 "${COSMOS3_ONNX_SRC_DIR}/modify_dopt_config.py" \
    "${COSMOS3_QUANT_CONFIG}" \
    --bit "${COSMOS3_QUANT_BIT}" \
    --report-path "${COSMOS3_QUANT_CONFIG_REPORT}"

echo "========== Step 3: calibrate and generate quant files =========="
python3 "${COSMOS3_ONNX_SRC_DIR}/cosmos3_quantize.py" \
    --stage quant \
    --quant-output-dir "${COSMOS3_QUANT_OUTPUT_DIR}" \
    --embedding-separate "${COSMOS3_EMBEDDING_SEPARATE}" \
    "${COMMON_ARGS[@]}"

echo "========== Step 4: export DOPT fake-quant ONNX =========="
python3 "${COSMOS3_ONNX_SRC_DIR}/convert_cosmos3_to_onnx.py" \
    --output "${COSMOS3_ONNX_RAW}" \
    --exporter "${COSMOS3_ONNX_EXPORTER}" \
    --external-data-mode "${COSMOS3_EXTERNAL_DATA_MODE}" \
    "${COMMON_ARGS[@]}"

echo "========== Step 5: apply Cosmos3 Policy compatibility rewrites =========="
if ! python3 scripts/rewrite_action_policy_onnx.py \
    "${COSMOS3_ONNX_RAW}" \
    "${COSMOS3_ONNX_COMPATIBLE}" \
    --prompt-embedding-table-path "${COSMOS3_PROMPT_EMBEDDING}" \
    --report-path "${COSMOS3_REWRITE_REPORT}" \
    --verify-equivalence \
    --verification-provider CUDAExecutionProvider \
    --verification-provider CPUExecutionProvider
then
    rm -f -- \
        "${COSMOS3_ONNX_COMPATIBLE}" \
        "${COSMOS3_REWRITE_REPORT}" \
        "${COSMOS3_PROMPT_EMBEDDING}"
    exit 1
fi

echo "========== Step 6: normalize and deduplicate ONNX node names =========="
python3 "${COSMOS3_ONNX_SRC_DIR}/finalize_onnx.py" \
    --input "${COSMOS3_ONNX_COMPATIBLE}" \
    --output "${COSMOS3_ONNX_FINAL}" \
    --report-path "${COSMOS3_FINALIZE_REPORT}"

echo "========== Step 7: compare quant-file entries with ONNX nodes =========="
if [[ ! -f "${COSMOS3_QUANT_PARAMS_FILE}" ]]; then
    echo "Quant params file not found: ${COSMOS3_QUANT_PARAMS_FILE}" >&2
    exit 1
fi
python3 "${COSMOS3_ONNX_SRC_DIR}/match_quant_params.py" \
    --onnx "${COSMOS3_ONNX_FINAL}" \
    --quant-params "${COSMOS3_QUANT_PARAMS_FILE}" \
    --output-json "${COSMOS3_MATCH_REPORT}"

echo "========== Step 8: strict final ONNX audit =========="
python3 "${COSMOS3_ONNX_SRC_DIR}/audit_onnx.py" \
    "${COSMOS3_ONNX_FINAL}" \
    --max-rank 4 \
    --report-path "${COSMOS3_AUDIT_REPORT}"

echo "========== Step 9: remove reproducible intermediate files =========="
rm -f -- \
    "${COSMOS3_ONNX_RAW}" \
    "${COSMOS3_ONNX_COMPATIBLE}" \
    "${COSMOS3_QUANT_CONFIG}.backup" \
    "${COSMOS3_QUANT_ROOT}/quant_info.txt" \
    "${COSMOS3_REWRITE_REPORT}" \
    "${COSMOS3_FINALIZE_REPORT}"

echo "========== Done =========="
echo "DOPT config: ${COSMOS3_QUANT_CONFIG}"
echo "fake-quant weight: ${COSMOS3_FAKEQUANT_WEIGHT}"
echo "quant params: ${COSMOS3_QUANT_PARAMS_FILE}"
echo "final ONNX: ${COSMOS3_ONNX_FINAL}"
echo "ONNX external data: ${COSMOS3_ONNX_RAW}.data"
echo "prompt embedding table: ${COSMOS3_PROMPT_EMBEDDING}"
echo "name match report: ${COSMOS3_MATCH_REPORT}"
echo "audit report: ${COSMOS3_AUDIT_REPORT}"
