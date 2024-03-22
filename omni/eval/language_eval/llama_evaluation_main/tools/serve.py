#!/usr/bin/env python3

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from omni.utils.loguru import logger

from llama_evaluation.tasks.codegen_math_online import online_eval, online_parser
from llama_evaluation.tasks.multich import multich_online_eval, multich_online_parser

app = FastAPI()


class Params(BaseModel):
    model_ip: str
    workers: int = 1
    model_name: str = None
    task: str = "codegen,math"
    to_db: bool = False
    max_gen_length: int = 512


@app.post("/run_benchmark")
async def run_benchmark(data: Params, background_tasks: BackgroundTasks):
    """start the benchmark service, non-blocking"""

    def merge_args(args, data, task: str):
        args.addr = data.model_ip
        args.model = data.model_name
        args.no_db = not data.to_db
        args.max_gen_length = data.max_gen_length
        args.task = task
        args.jsonl_file = f"{task}_samples.jsonl"
        return args

    def benchmark_task():
        # Define the time-consuming task here
        tasks = data.task.split(",")
        print(f"benchmarking tasks: {tasks}")
        if "codegen" in tasks:
            args = online_parser().parse_known_args()[0]
            args = merge_args(args, data, "codegen")
            tasks.remove("codegen")
            online_eval(args)
        if "math" in tasks:
            args = online_parser().parse_known_args()[0]
            args = merge_args(args, data, "math")
            tasks.remove("math")
            online_eval(args)
        if tasks:
            all_tasks = ",".join(tasks)
            args = multich_online_parser().parse_known_args()[0]
            args = merge_args(args, data, all_tasks)
            args.tasks = all_tasks
            args.mode = "online"
            multich_online_eval(args)

    # Run the long-running task in the background
    background_tasks.add_task(benchmark_task)
    return {"message": "Task started in the background."}
