#!/bin/bash

set -e


RED='\033[31m'
GREEN='\033[32m'
YELLOW='\033[33m'
NC='\033[0m' # No Color

function log() {
    TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}$TIME${NC}" "$1"
}

function log_warning() {
    TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}$TIME${NC}" "$1"
}

function log_error() {
    TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${RED}$TIME${NC}" "$1"
}

error_handler() {
    log_error "Error occurred in script at line: $1"
    log_error "Line exited with status: $2"
    exit $2
}

trap 'error_handler ${LINENO} $?' ERR

function helpFunction()
{
    echo ""
    echo "Usage: $0 --env_name=ENV_NAME --py_ver=PY_VER --cuda=CUDA --torch_dir=TORCH_DIR"
    echo "Example: $0 --env_name=dreamllm --py_ver=3.10 --cuda=118 --torch_dir=/data/torch-2.1.2/"
    echo -e "\t--env_name: conda environment name, default name is \`dreamllm\`"
    echo -e "\t--py_ver: python version, default version is 3.10"
    echo -e "\t--cuda: cuda version, used to install pytorch, default version is 118"
    echo -e "\t--torch_dir: directory address containing torch whl, if specified, the cuda version will be ignored"
    echo -e "\t-h or --help: show help information"
    echo ""
}

for i in "$@"
do
case $i in
    --env_name=*)
    ENV_NAME="${i#*=}"
    shift   # past argument=value
    ;;
    --py_ver=*)
    PY_VER="${i#*=}"
    shift   # past argument=value
    ;;
    --cuda=*)
    CUDA="${i#*=}"
    shift   # past argument=value
    ;;
    --torch_dir=*)
    TORCH_DIR="${i#*=}"
    shift   # past argument=value
    ;;
    -h|--help)
    HELP=1
    shift   # past argument with no value
    ;;
    *)
            # unknown option
    ;;
esac
done

if [ $HELP ]
then
    helpFunction
    exit 0
fi

if [ ! "$ENV_NAME" ]
then
    ENV_NAME="dreamllm"
fi

if [ ! "$PY_VER" ]
then
    PY_VER="3.10"
fi

if [ ! "$CUDA" ]
then
    CUDA="118"
fi

if [ ! "$TORCH_DIR" ]
then
    TORCH_DIR=""
fi

log "Environment name  = ${ENV_NAME}"
log "Python version    = ${PY_VER}"
log "CUDA version      = ${CUDA}"
log "Torch directory   = ${TORCH_DIR}"

# Check if environment exists
if ! conda info --envs | grep "$ENV_NAME"; then
    log "Conda environment '$ENV_NAME' does not exist. Creating..."
    conda create -n "$ENV_NAME" python=$PY_VER -y
else
    log_warning "Conda environment '$ENV_NAME' already exists."
fi

# activate environment
source activate "$ENV_NAME"
log "Conda environment '$ENV_NAME' has been activated."

log "Installing torch ..."
if [ ! -z "$TORCH_DIR" ]; then
    # install torch
    for wheel in "$TORCH_DIR"/*.whl; do
        log "Installing $wheel ..."
        pip install "$wheel"
    done
elif [ ! -z "$CUDA" ]; then
    log_warning "If installing torch gets stuck here, please download the whl file in advance to a folder and then specify the directory."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu$CUDA
else
    log_error "Please specify the cuda version or the directory of torch whl files."
    exit 1
fi
log "torch has been installed."

log "Installing accelerate ..."
pip install accelerate==0.23.0 -c constraints.txt
log "accelerate has been installed."

# Install flash-attn
log "Installing flash-attn ..."
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
log "flash-attn has been installed."


# Install the current directory as a package
log "Installing the current directory as a package ..."
pip install -e .
log "The current directory has been installed as a package."


# Install every project in the third_party directory
log "Installing packages in third_party directory ..."
THIRD_PARTY_DIR="./third_party"
for project_dir in "$THIRD_PARTY_DIR"/*; do
    # check if it is a folder
    if [ -d "$project_dir" ]; then
        log "Installing package in $project_dir ..."
        pip install -e "$project_dir"
    fi
done
log "All packages in $THIRD_PARTY_DIR have been installed."

log "Successfully setup the environment."