#!/bin/bash

set -e
set -o pipefail

# Path detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
BEYONDAGENT_DIR="$(dirname "$(dirname "$ENV_SERVICE_DIR")")"
BFCL_ROOT="$SCRIPT_DIR"
WORKSPACE_DIR="$BEYONDAGENT_DIR"


# 2. 环境变量配置
echo "📁 设置环境变量..."
export NODE_ENV=production
export WORKSPACE_DIR="$WORKSPACE_DIR"
export BFCL_ROOT="$BFCL_ROOT"
export PYTHONPATH="$BEYONDAGENT_DIR:$PYTHONPATH"


BFCL_CONDA_ENV="${BFCL_CONDA_ENV:-bfcl}"

is_active_bfcl_env() {
    [ "${CONDA_DEFAULT_ENV:-}" = "$BFCL_CONDA_ENV" ]
}

conda_env_ready() {
    conda run -n "$BFCL_CONDA_ENV" python -V >/dev/null 2>&1
}

bfcl_run() {
    if is_active_bfcl_env; then
        "$@"
    else
        conda run -n "$BFCL_CONDA_ENV" "$@"
    fi
}

# 3. Conda 环境创建
if is_active_bfcl_env; then
    echo "✅ 当前已激活 Conda 环境 $BFCL_CONDA_ENV。"
elif conda_env_ready; then
    echo "✅ Conda 环境 $BFCL_CONDA_ENV 已可用。"
else
    if conda info --envs | awk '{print $1}' | grep -x "$BFCL_CONDA_ENV" >/dev/null 2>&1; then
        echo "❌ Conda 环境 $BFCL_CONDA_ENV 存在但 conda run 无法使用。"
        echo "请先执行："
        echo "  conda activate $BFCL_CONDA_ENV"
        echo "  bash $SCRIPT_DIR/setup.sh"
        exit 1
    fi
    echo "🐍 创建 Conda 环境 $BFCL_CONDA_ENV（Python 3.11.13）..."
    conda create -n "$BFCL_CONDA_ENV" python=3.11.13 -y
fi

# 4. 安装依赖
cd "$SCRIPT_DIR"
if [ -d "$SCRIPT_DIR/gorilla" ]; then
    echo "🔄 更新 gorilla 仓库..."
    cd "$SCRIPT_DIR/gorilla"
    git pull
else
    echo "📦 克隆 gorilla 仓库..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/ShishirPatil/gorilla.git
fi

echo "📋 安装 Python 依赖..."

bfcl_run pip install -e "$SCRIPT_DIR/gorilla/berkeley-function-call-leaderboard/."
bfcl_run pip install -r "$SCRIPT_DIR/requirements.txt"

# 5. 准备数据
echo "📁 准备 BFCL 数据..."
cp -r "$SCRIPT_DIR/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data" "$SCRIPT_DIR/bfcl_eval"

cd "$SCRIPT_DIR/"
echo "当前工作目录: $(pwd)"
echo "脚本目录: $SCRIPT_DIR"

bfcl_run python "$SCRIPT_DIR/bfcl_dataprocess.py"

# 6. 设置环境变量
echo "🌎 设置环境变量..."
export ENV_PATH="$SCRIPT_DIR"
export BFCL_DATA_PATH="$ENV_PATH/bfcl_data/multi_turn_processed.jsonl"
export BFCL_SPLID_ID_PATH="$WORKSPACE_DIR/data/bfcl_400_split.json"
export BFCL_ANSWER_PATH="$ENV_PATH/bfcl_eval/possible_answer"
export OPENAI_API_KEY="$OPENAI_API_KEY"

echo "✅ 设置完成！"

echo ""


echo "如果需要更换数据集，可以使用以下命令："
echo ""
echo "可选的数据集有:"
echo "- all"
echo "- all_scoring"
echo "- multi_turn"
echo "- single_turn"
echo "- live"
echo "- non_live"
echo "- non_python"
echo "- python"
echo "- multi_turn_base"
echo ""
echo "示例命令:"
echo "export DATASET_NAME=multi_turn"
echo "export BFCL_DATA_PATH=\"$ENV_PATH/bfcl_data/\${DATASET_NAME}_processed.jsonl\""
echo "export BFCL_SPLID_ID_PATH=\"$ENV_PATH/bfcl_data/\${DATASET_NAME}_split_ids.json\""
echo ""
echo "将 \$DATASET_NAME 替换为您想要使用的数据集名称即可。"

echo "👉 启动方法："
echo "----------------------------------------"
echo "source \$(conda info --base)/etc/profile.d/conda.sh"
echo "conda activate bfcl"
echo "cd $BEYONDAGENT_DIR/env_service/launch_script"
echo "bash bfcl.sh"
echo "----------------------------------------"
