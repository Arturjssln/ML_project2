dirname="$(dirname ${BASH_SOURCE[0]})/"
vars_path="$dirname""conda_vars.sh"

source $vars_path

if conda env list|grep -q "$ENV_NAME"; then
    echo "conda env $ENV_NAME already exists: updating..."
    "$dirname""update_env.sh"
else
    echo "conda env $ENV_NAME doesn't exist: creating..."
    "$dirname""create_env.sh"
fi