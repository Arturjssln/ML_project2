dirname="$(dirname ${BASH_SOURCE[0]})/"
vars_path="$dirname""conda_vars.sh"

source $vars_path

env_path="$dirname$ENV_PATH"

set -x
conda env create -n "$ENV_NAME" -f "$env_path"
set +x