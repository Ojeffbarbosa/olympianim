#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
app_home="${OLYMPIANIM_HOME:-$script_dir}"
workspace_dir="$app_home/workspace"

if [[ "$app_home" == "/" || -z "$app_home" ]]; then
    echo "Erro: diretório do aplicativo inválido: '$app_home'." >&2
    exit 1
fi

if [[ ! -f "$script_dir/pyproject.toml" ]]; then
    echo "Erro: execute este script a partir de uma instalação válida do Olympianim." >&2
    exit 1
fi

echo "ATENÇÃO: esta operação apagará permanentemente:"
echo "  - banco de dados e configurações em: $workspace_dir"
echo "  - todos os projetos, códigos, áudios, vídeos e logs"
echo
echo "O código-fonte, o ambiente virtual e os arquivos .env e .env.example serão preservados."
echo "Feche o Olympianim antes de continuar."
echo
read -r -p "Digite RESETAR para confirmar: " confirmation

if [[ "$confirmation" != "RESETAR" ]]; then
    echo "Reset cancelado; nenhum arquivo foi removido."
    exit 0
fi

rm -rf -- "$workspace_dir"
mkdir -p -- "$workspace_dir/projects"
touch -- "$workspace_dir/projects/.gitkeep"

echo "Olympianim resetado com sucesso."
echo "Na próxima execução, o banco de dados e os prompts padrão serão recriados."
