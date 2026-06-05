#!/usr/bin/env bash
set -e

# 1. 環境変数の設定 (念のため)
export PATH="/usr/local/bin:/home/vscode/.local/bin:/home/vscode/.npm-global/bin:$PATH"

# 2. uv プロジェクトの初期化 (pyproject.toml がない場合のみ)
if [ ! -f "pyproject.toml" ]; then
    echo "🚀 Initializing a new uv project..."
    uv init --no-workspace .
    uv add setuptools wheel jupyterlab ipykernel
else
    echo "✅ pyproject.toml already exists."
    uv sync
fi

# 3. Jupyter kernel の登録
#echo "🛠️ Setting up Jupyter kernel..."
#uv run python -m ipykernel install --user --name "uv" --display-name "uv(repo)" || true

# 4. npm ツールのインストール
mkdir -p /home/vscode/.npm-global
npm config set prefix /home/vscode/.npm-global
echo "🛠️ Installing global npm tools..."
npm install -g @marp-team/marp-cli @mermaid-js/mermaid-cli

echo "✨ All processes finished successfully!"