#!/usr/bin/env bash

function log() {
    TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "\033[32m$TIME\033[0m" "$@"
}

function warn_log() {
    TIME=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "\033[32m$TIME\033[0m" "\033[31m$@\033[0m"
}

function helpFunction()
{
    echo ""
    echo "Usage: $0 --task=xx --model-size=xx --ckpt-dir=xx --temperature=xx"
    echo "Example: $0 --task=codegen --model-size=7B --ckpt-dir=/path/to/ckpt"
    echo -e "\t--task: task name to benchmark, default is all tasks, options are [all, codegen, math, multich]"
    echo -e "\t--model-size: size of model, e.g. 7B/13B/65B"
    echo -e "\t--ckpt-dir: dir of check point, if not given tensorRT will not be installed"
    echo -e "\t--temperature: model temperature. default is 0.1"
    echo -e "\t--code-batch: batchsize of code. default is 32"
    echo -e "\t--math-batch: batchsize of math. default is 16"
    echo -e "\t--origin-llama: benchmark origin llama model"
    echo -e "\t-h or --help: show help information"
}

function benchmark_task()
{
    local task=$1 model_size=$2 ckpt_dir=$3 temperature=$4 code_batch=$5 math_batch=$6 load_orgin=$7
    local other_args="${@:8}"

    local ckpt_path
    if [ "$load_orgin" -eq 1 ]
    then
        ckpt_path=path2model/weights/$ckpt_dir
    else
        ckpt_path=path2model/checkpoints/$ckpt_dir
    fi

    local tokenizer_path
    if [ "$load_orgin" -eq 1 ]
    then
        tokenizer_path=path2model/weights/$ckpt_dir/tokenizer.model
    else
        tokenizer_path=path2model
    fi

    if [ "$task" == "codegen" ]
    then
        benchmark_codegen $model_size $ckpt_path $temperature $code_batch $tokenizer_path $other_args
    elif [ "$task" == "math" ]
    then
        benchmark_math $model_size $ckpt_path $temperature $math_batch $tokenizer_path $other_args
    elif [ "$task" == "multich" ]
    then
        benchmark_multich $model_size $ckpt_path $temperature $tokenizer_path
    elif [ "$task" == "all" ]
    then
        benchmark_codegen $model_size $ckpt_path $temperature $code_batch $tokenizer_path $other_args
        benchmark_math $model_size $ckpt_path $temperature $math_batch $tokenizer_path $other_args
        benchmark_multich $model_size $ckpt_path $temperature $tokenizer_path
    else
        warn_log "Unknown task $1"
        exit 1
    fi
}


function benchmark_codegen()
{
    local model_size=$1 ckpt_path=$2 temperature=$3 code_batch=$4 tokenizer_path=$5
    local other_args="${@:6}"

    torchrun --nproc_per_node $MLP_WORKER_GPU llama_evaluation/tasks/codegen.py \
        -n 1 -k 1 \
        --model_size "$model_size" --ckpt_dir $ckpt_path \
        --tokenizer_path $tokenizer_path \
        --temperature "$temperature" \
        --batch_size $code_batch $other_args
}

function benchmark_math()
{
    local model_size=$1 ckpt_path=$2 temperature=$3 math_batch=$4 tokenizer_path=$5
    local other_args="${@:6}"

    torchrun --nproc_per_node $MLP_WORKER_GPU llama_evaluation/tasks/math_eval.py \
        --num-samples 1 --model_size "$model_size" \
        --ckpt_dir $ckpt_path \
        --tokenizer_path "$tokenizer_path" \
        --temperature "$temperature" \
        --batch_size $math_batch $other_args
        # --max_seq_len 4096  \
}

function benchmark_multich()
{
    tmp=$(echo $ckpt_path | grep "sft")
    local model_size=$1 ckpt_path=$2 temperature=$3 tokenizer_path=$4
    torchrun --nproc_per_node "$MLP_WORKER_GPU" llama_evaluation/tasks/multich.py \
        --model_size "$model_size" \
        --ckpt_dir "$ckpt_path" \
        --tokenizer_path "$tokenizer_path" \
        --temperature "$temperature" \
        --lite $other_args
}

function main()
{
    for i in "$@"
    do
    case $i in
        --task=*)
        TASK="${i#*=}"
        shift # past argument=value
        ;;
        --model-size=*)
        MODEL_SIZE="${i#*=}"
        shift # past argument=value
        ;;
        --ckpt-dir=*)
        CKPT_DIR="${i#*=}"
        shift # past argument=value
        ;;
        --temperature*)
        TEMPERATURE="${i#*=}"
        shift # past argument=value
        ;;
        --code-batch=*)
        CODE_BATCH="${i#*=}"
        shift # past argument=value
        ;;
        --math-batch=*)
        MATH_BATCH="${i#*=}"
        shift # past argument=value
        ;;
        --origin-llama)
        LOAD_ORIGIN=1
        shift # past argument with no value
        ;;
        -h|--help)
        HELP=1
        shift # past argument with no value
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

    if [ ! "$TASK" ]
    then
        TASK="all"
    fi

    if [ ! "$TEMPERATURE" ]
    then
        TEMPERATURE="0.1"
    fi

    if [ ! "$CODE_BATCH" ]
    then
        CODE_BATCH=32
    fi

    if [ ! "$MATH_BATCH" ]
    then
        MATH_BATCH=16
    fi

    if [ ! "$LOAD_ORIGIN" ]
    then
        LOAD_ORIGIN=0
    fi

    benchmark_task $TASK $MODEL_SIZE $CKPT_DIR $TEMPERATURE $CODE_BATCH $MATH_BATCH $LOAD_ORIGIN ${@:1}
}

main $@
