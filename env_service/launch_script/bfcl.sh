#!/bin/bash
set -eu

# 注意：
# 请先执行文件：EnvService/env_sandbox/environments/bfcl/bfcl_dataprocess.py
# 获取BFCL_DATA_PATH与BFCL_SPLID_ID_PATH，并对应设置以上两个变量

# 环境变量（请改成实际路径）

#
# 获取 launch_script 的目录
LAUNCH_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取 env_service 的目录
ENV_SERVICE_DIR="$(dirname "$LAUNCH_SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$LAUNCH_SCRIPT_DIR/../../" && pwd)"

# 获取 bfcl 环境的目录
DEFAULT_BFCL_ENV_DIR="$ENV_SERVICE_DIR/environments/bfcl"
BFCL_ENV_DIR=${BFCL_ENV_DIR:-$DEFAULT_BFCL_ENV_DIR}
DEFAULT_BFCL_DATA_PATH="$BFCL_ENV_DIR/bfcl_data/multi_turn_processed.jsonl"
DEFAULT_BFCL_SPLID_ID_PATH="$PROJECT_ROOT/data/bfcl_400_split.json"

export ENV_PATH="$BFCL_ENV_DIR"
export BFCL_DATA_PATH="${BFCL_DATA_PATH:-$DEFAULT_BFCL_DATA_PATH}"
export BFCL_SPLID_ID_PATH="${BFCL_SPLID_ID_PATH:-${BFCL_SPLIT_ID_PATH:-$DEFAULT_BFCL_SPLID_ID_PATH}}"
export BFCL_ANSWER_PATH="${BFCL_ANSWER_PATH:-$BFCL_ENV_DIR/bfcl_eval/possible_answer}"
export BFCL_HOST="${BFCL_HOST:-127.0.0.1}"
export BFCL_PORT="${BFCL_PORT:-8082}"

echo "🌍 环境变量设置完成:"
echo "ENV_PATH: $ENV_PATH"
echo "BFCL_DATA_PATH: $BFCL_DATA_PATH"
echo "BFCL_SPLID_ID_PATH: $BFCL_SPLID_ID_PATH"
echo "BFCL_ANSWER_PATH: $BFCL_ANSWER_PATH"
echo "BFCL_HOST: $BFCL_HOST"
echo "BFCL_PORT: $BFCL_PORT"

# 检查文件是否存在
missing_required=0

if python -c "import bfcl_eval" >/dev/null 2>&1; then
    echo "✅ bfcl_eval Python package is importable"
else
    echo "❌ bfcl_eval Python package is not importable in the current environment"
    missing_required=1
fi

if [ -f "$BFCL_DATA_PATH" ]; then
    echo "✅ 数据文件存在: $BFCL_DATA_PATH"
else
    echo "❌ 数据文件不存在: $BFCL_DATA_PATH"
    missing_required=1
fi

if [ -f "$BFCL_SPLID_ID_PATH" ]; then
    echo "✅ 分割ID文件存在: $BFCL_SPLID_ID_PATH"
else
    echo "❌ 分割ID文件不存在: $BFCL_SPLID_ID_PATH"
    missing_required=1
fi

if [ -d "$BFCL_ANSWER_PATH" ]; then
    echo "✅ 答案文件夹存在: $BFCL_ANSWER_PATH"
else
    echo "❌ 答案文件夹不存在: $BFCL_ANSWER_PATH"
    missing_required=1
fi

if [ "$missing_required" -ne 0 ]; then
    cat <<EOF

BFCL environment is incomplete. Run setup once before launching BFCL:

  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate bfcl
  bash env_service/environments/bfcl/setup.sh

After setup completes, run:

  conda activate bfcl
  python -c "import bfcl_eval; print('bfcl_eval ok')"
  test -f "$BFCL_DATA_PATH"
  test -d "$BFCL_ANSWER_PATH"

EOF
    exit 1
fi

export OPENAI_API_KEY=xx

# only for multinode running
export RAY_ENV_NAME=bfcl 

cd "$PROJECT_ROOT"

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# 打印当前工作目录和 PYTHONPATH 以进行调试
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

# 运行 Python 命令
exec python -m env_service.env_service --env bfcl --portal "$BFCL_HOST" --port "$BFCL_PORT"
