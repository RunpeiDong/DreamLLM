#!/usr/bin/env python3

import os
import time
import sqlite3
import argparse
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple, Dict
from omni.utils.loguru import logger

from llama_evaluation_main.llama_evaluation.utils.task_utils import BASE_TASKS
from llama_evaluation_main.llama_evaluation.utils.math_utils import get_answer_str

__all__ = [
    "DATABASE_PATH", "HEADERS", "DB_NAME",
    "py_type2db_type", "create_database_table", "delete_database_table",
    "add_columns_into_table", "merge_table", "db_str",
    "where_clause_str", "setval_clause_str", "select_from_db",
    "delete_table_content", "check_values", "insert_data",
    "update_data", "delete_data", "fill_dataset_table",
    "fill_datasets_info_table", "delete_table", "create_table",
    "display_data", "add_db_columns", "read_db_content",
    "update_db_content", "delete_db_content", "check_field",
    "unpack_data", "get_dataset_info", "get_model_info",
    "get_topk_modelid", "get_max_eval_count", "get_metric_values",
    "ensure_model_info_exist", "create_all_database", "fill_evaluate_metrics",
    "metrics_to_database", "write_metrics_by_args", "all_evaluate_model_id",
]

DATABASE_PATH = os.environ.get("EVAL_DATABASE_PATH", "data/database/")


HEADERS = {
    # model info: store info of model such as temperature, topk, etc.
    "model_info": ["id", "model_name", "temperature", "topp", "topk"],
    "model_info_type": [int, str, float, float, float],
    # datatset info is used to store info of dataset such as type, language, etc.
    "datasets_info": ["id", "dataset", "task_type", "language"],
    "datasets_info_type": [int, str, str, str],

    # evaluation info: store info of evaluation, one table per task_type
    "codegen_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output", "pass", "function"],
    "codegen_eval_type": [int, int, int, int, int, str, str, str],
    "generation_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output","exact_match", "include", "raw_output"],  # noqa
    "generation_eval_type": [int, int, int, int, int, str, str, str, str],
    "multi_choice_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output", "logprobs", "pass"],
    "multi_choice_eval_type": [int, int, int, int, int, int, str, str],
    "math_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output", "short_answer", "pass"],
    "math_eval_type": [int, int, int, int, int, str, str, str],
    "bbh_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output","exact_match","raw_output"],  # noqa
    "bbh_eval_type": [int, int, int, int, int, str, str, str],
    "ceval_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output", "logprobs", "pass"],
    "ceval_eval_type": [int, int, int, int, int, str, str, str],
    "mmlu_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output", "logprobs", "pass"],
    "mmlu_eval_type": [int, int, int, int, int, str, str, str],
    "agieval_eval": ["id", "model_id", "dataset_id", "eval_count", "sample_id", "output", "logprobs", "pass"],
    "agieval_eval_type": [int, int, int, int, int, str, str, str],

    # metric info
    "evaluate_metrics": ["id", "model_id", "dataset_id", "eval_count", "metric_type", "metric_value"],
    "evaluate_metrics_type": [int, int, int, int, str, float],

    # dataset content: store info of single dataset
    "human_eval": ["id", "question", "test"],
    "human_eval_type": [int, str, str],
    "math": ["id", "problem", "level", "type", "solution", "short_solution"],
    "math_type": [int, str, str, str, str, str],
    # multiple choice QA
    "boolq": ["id", "question", "options", "label"],
    "piqa": ["id", "question", "options", "label"],
    "siqa": ["id", "question", "options", "label"],
    "sciq": ["id", "question", "options", "label"],
    "hellaswag": ["id", "question", "options", "label"],
    "winogrande": ["id", "question", "options", "label"],
    "arc_e": ["id", "question", "options", "label"],
    "arc_c": ["id", "question", "options", "label"],
    "obqa": ["id", "question", "options", "label"],
    "race_m": ["id", "question", "options", "label"],
    "race_h": ["id", "question", "options", "label"],
    "clue_c3": ["id", "question", "options", "label"],
    "clue_wsc": ["id", "question", "options", "label"],
    # short generation
    "triviaqa": ["id", "question", "labels"],
    "naturalqa": ["id", "question", "labels"],
    "drop_gen": ["id", "question", "labels"],
    "clue_cmrc": ["id", "question", "labels"],
    "xtreme": ["id", "question", "labels"],
}

for dataset_name in BASE_TASKS:
    if "bbh" in dataset_name or "ceval" in dataset_name or "mmlu" in dataset_name or "agieval" in dataset_name:
        HEADERS[dataset_name] = ["id", "question", "labels"]

DB_NAME = {
    "model_info": "model_info.sqlite",
    "datasets_info": "dataset_info.sqlite",
    # evaluation database, one database per task
    "codegen_eval": "evaluate_info.sqlite",
    "generation_eval": "evaluate_info.sqlite",
    "multi_choice_eval": "evaluate_info.sqlite",
    "math_eval": "evaluate_info.sqlite",
    "bbh_eval": "evaluate_info.sqlite",
    "ceval_eval": "evaluate_info.sqlite",
    "mmlu_eval": "evaluate_info.sqlite",
    "agieval_eval": "evaluate_info.sqlite",
    # dataset db file, all in one
    "human_eval": "dataset.sqlite",
    "math": "dataset.sqlite",
    "evaluate_metrics": "metrics.sqlite",
}
for dataset_name in BASE_TASKS:
    DB_NAME[dataset_name] = "dataset.sqlite"
del dataset_name


def py_type2db_type(pytype: List) -> List[str]:
    map_dict = {
        int: "INTEGER",
        str: "TEXT",
        float: "REAL",
        # BLOB, NULL, NUMERIC
    }
    db_type = [map_dict[x] for x in pytype]
    db_type[0] = db_type[0] + " PRIMARY KEY AUTOINCREMENT"
    return db_type


def parse_str(x: str) -> str:
    return x.replace("/", "__")


def create_database_table(
    db_file: str, table: str, headers: List[str], types: List[str]
) -> sqlite3.Connection:
    """
    open a database file and create a table if not exist according to headers and types

    Args:
        db_file (str): file name of database to create table.
        table (str):
        headers (List[str]):
        types (List[str]):
    """
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    cursor = conn.cursor()

    cmd = f"CREATE TABLE IF NOT EXISTS {parse_str(table)} ({', '.join([f'{h} {t}' for h, t in zip(headers, types)])})"  # noqa
    cursor.execute(cmd)
    cursor.close()
    return conn


def delete_database_table(db_file: str, table: str):
    """delete table from database if exists"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    cmd = f"DROP TABLE IF EXISTS {parse_str(table)}"
    cursor.execute(cmd)
    return conn


def add_columns_into_table(conn: sqlite3.Connection, table: str, headers: List[str], types: List[str]):
    """add columns into database table.

    Args:
        conn (sqlite3.Connection): connection to database
        table (str): table name
        headers (List[str]): column names
        types (List[str]): column types
    """
    cursor = conn.cursor()
    for h, t in zip(headers, types):
        cmd = f"ALTER TABLE {parse_str(table)} ADD COLUMN {h} {t}"
        cursor.execute(cmd)
    cursor.close()
    conn.commit()


def merge_table(src_db_file: str, dst_db_file: str, table: str, dst_table: str = None):
    """drop table from src database if exists and create a new one with same name and schema in dst database"""
    if dst_table is None:
        dst_table = table

    src_conn = sqlite3.connect(src_db_file)
    src_cursor = src_conn.cursor()
    data = select_from_db(src_cursor, table)
    data_type = [type(x) for x in data[0]]
    col_names = [x[0] for x in src_cursor.description]
    conn = create_database_table(dst_db_file, dst_table, col_names, py_type2db_type(data_type))
    for d in data:
        insert_data(conn, dst_table, d, auto_inc=False)
    conn.commit()
    conn.close()


def db_str(x: str) -> str:
    if isinstance(x, str):
        return x.replace("'", "''")
    return x


def where_clause_str(**kwargs) -> str:
    where_clause = ""
    if kwargs:
        where_clause = "WHERE "
        where_stats = [f"{k}='{db_str(v)}'" for k, v in kwargs.items()]
        where_clause += " AND ".join(where_stats)
    return where_clause


def setval_clause_str(**kwargs) -> str:
    update_clause = ""
    if kwargs:
        update_clause = "SET "
        update_stats = [f"{k}='{db_str(v)}'" for k, v in kwargs.items()]
        update_clause += ", ".join(update_stats)
    return update_clause


def select_from_db(cursor: sqlite3.Cursor, table_name, field="all", **kwargs) -> List:
    select_info = "*" if field == "all" else field
    if isinstance(field, (List, Tuple)):
        select_info = ", ".join(field)

    where_clause = where_clause_str(**kwargs)
    cursor.execute(f"SELECT {select_info} FROM {parse_str(table_name)} {where_clause}")
    data = cursor.fetchall()
    return data


def delete_table_content(cursor: sqlite3.Cursor, table_name, **kwargs):
    cmd = f"DELETE FROM {parse_str(table_name)} {where_clause_str(**kwargs)}"
    cursor.execute(cmd)


def check_values(cursor, table_name, values, many=True, auto_inc=True, keys2check=None):
    cursor = cursor.execute(f"SELECT * FROM {parse_str(table_name)}")
    col_name = [x[0] for x in cursor.description]
    if auto_inc:
        col_name.pop(0)  # pop primary key since it is autoincrement

    if keys2check is None:
        keys2check = col_name

    def value_exist(value) -> bool:
        kv_pair = {k: v for k, v in zip(col_name, value) if k in keys2check}
        data = select_from_db(cursor, table_name, **kv_pair)
        return data

    if many:
        return [x for x in values if not value_exist(x)]
    else:
        return [] if value_exist(values) else values


def insert_data(
    connection: sqlite3.Connection, table: str, values: List,
    many: bool = False, auto_inc: bool = True, check: bool = True, keys2check=None,
    max_retry_count: int = 5, second_between_retry: int = 1,
):
    """Insert data into a database table. It's a idempotent function.

    Args:
        connection (sqlite3.Connection): connection to a database.
        table (str): table name.
        values (List): values to insert.
        many (bool): whether to insert many data at once. Default: False.
        auto_inc (bool): whether to insert autoincrement id. Default: True.
        check (bool): whether to check if the values already exist. Default: True.
        keys2check (List[str]): keys to check if the values already exist.
            If the value is None, all values to insert will be checked.  Default: None.
        max_retry_count (int): maximum retry count. Set to 1 if retry is not needed. Default: 5.
        second_between_retry (int): time delta between next retry in second. Default: 1.
    """
    # TODO: check if the values already exist
    cursor = connection.cursor()
    values_str = ", ".join(["?" for _ in values])
    if auto_inc:
        values_str = "NULL, " + values_str

    if check:
        values = check_values(
            cursor, table, values, many=many, auto_inc=auto_inc, keys2check=keys2check
        )
        if not values:
            return

    cmd = f"INSERT INTO {parse_str(table)} VALUES ({values_str})"
    retry_count = 0
    max_retry_count = max(max_retry_count, 1)  # ensure try at least once
    while retry_count < max_retry_count:
        try:
            if many:
                cursor.executemany(cmd, values)
            else:
                cursor.execute(cmd, values)
            break
        except Exception:
            retry_count += 1
            print(f"Retrying {retry_count} times...")
            time.sleep(second_between_retry)

    connection.commit()


def update_data(connection: sqlite3.Connection, table: str, update_dict: Dict, close=True, **kwargs):
    cursor = connection.cursor()
    data = select_from_db(cursor, table, **kwargs)
    assert len(data) == 1, "only one row should be updated"
    update_clause = setval_clause_str(**update_dict)
    where_clause = where_clause_str(**kwargs)
    cursor.execute(f"UPDATE {parse_str(table)} {update_clause} {where_clause}")
    connection.commit()
    if close:
        connection.close()
    else:
        return connection


def delete_data(connection: sqlite3.Connection, table: str, close=True, **kwargs):
    cursor = connection.cursor()
    delete_table_content(cursor, table, **kwargs)
    connection.commit()
    if close:
        connection.close()
    else:
        return connection


def fill_dataset_table(table_name="human_eval"):
    if table_name == "human_eval":
        db_file, headers = DB_NAME[table_name], HEADERS[table_name]
        types = py_type2db_type(HEADERS[table_name + "_type"])
        assert len(headers) == len(types), "headers and types should have the same length"
        dataset = load_dataset("openai_humaneval", split="test")

        conn = create_database_table(db_file, table_name, headers, types)
        data = select_from_db(conn.cursor(), table_name, id=1)
        if data:
            return
        for data in tqdm(dataset):
            task_id, prompt, test = data["task_id"], data["prompt"], data["test"]
            prime_id = int(task_id.split("/")[-1])
            data2db = (prime_id, prompt, test)
            insert_data(conn, table_name, data2db, auto_inc=False, keys2check=["id"])

    elif table_name == "math":
        db_file, headers = DB_NAME[table_name], HEADERS[table_name]
        types = py_type2db_type(HEADERS[table_name + "_type"])
        assert len(headers) == len(types), "headers and types should have the same length"

        dataset = load_dataset("competition_math", split="test")
        conn = create_database_table(db_file, table_name, headers, types)
        data = select_from_db(conn.cursor(), table_name, id=1)
        if data:
            return
        for data in tqdm(dataset):
            level, solution = data["level"], data["solution"]  # noqa
            level = int(level.split()[-1])
            answer_str = get_answer_str(solution)
            data2db = (data["problem"], level, data["type"], solution, answer_str)
            insert_data(conn, table_name, data2db, auto_inc=True, keys2check=["problem"])

    elif table_name in BASE_TASKS:
        task_type = get_dataset_info(table_name, field="task_type")
        db_file, headers = DB_NAME[table_name], HEADERS[table_name]
        if task_type == "multi_choice":
            types = py_type2db_type([int, str, str, int])
        else:
            types = py_type2db_type([int, str, str])
        assert len(headers) == len(types), "headers and types should have the same length"

        conn = create_database_table(db_file, table_name, headers, types)
        data = select_from_db(conn.cursor(), table_name, id=1)
        if data:
            return
        from llama_evaluation.data.datasets import load_dataset_by_name
        dataset = load_dataset_by_name(table_name, 0, None)

        for data in tqdm(dataset):
            if task_type == "generation":
                labels = data[1][:len(data[1])-data[1].count("NAN")]
                data2db = (data[0], str(labels))
            elif task_type == "multi_choice":
                data2db = (data[0], str(data[1]), data[2])
            else:
                data2db = (data[0], str(data[1]))
            insert_data(conn, table_name, data2db, auto_inc=True)

    else:
        raise NotImplementedError

    conn.commit()
    conn.close()


def fill_datasets_info_table(table_name="datasets_info"):
    db_file, headers = DB_NAME[table_name], HEADERS[table_name]
    types = py_type2db_type(HEADERS["datasets_info_type"])
    assert len(headers) == len(types), "headers and types should have the same length"

    type_info = [
        # codegen
        ("human_eval", "codegen", "en"),
        # generation
        ("math", "math", "en"),
        ("triviaqa", "generation", "en"),
        ("naturalqa", "generation", "en"),
        ("drop_gen", "generation", "en"),
        # multi_choice
        ("mmlu", "mmlu", "en"),
        ("boolq", "multi_choice", "en"),
        ("piqa", "multi_choice", "en"),
        ("siqa", "multi_choice", "en"),
        ("hellaswag", "multi_choice", "en"),
        ("winogrande", "multi_choice", "en"),
        ("arc_e", "multi_choice", "en"),
        ("arc_c", "multi_choice", "en"),
        ("obqa", "multi_choice", "en"),
        ("race_m", "multi_choice", "en"),
        ("race_h", "multi_choice", "en"),
        # cn dataset
        ("clue_wsc", "multi_choice", "cn"),
        ("clue_c3", "multi_choice", "cn"),
        ("clue_cmrc", "generation", "cn"),
        ("ceval", "ceval", "cn"),
        ("sciq", "multi_choice", "en"),
        ("xtreme", "generation", "cn"),
        ("agieval", "agieval", "cn"),
        ("bbh", "bbh", "en"),
    ]
    type_info.extend([(i, "bbh", "en") for i in BASE_TASKS if "bbh" in i])
    type_info.extend([(i, "ceval", "cn") for i in BASE_TASKS if "ceval" in i])
    type_info.extend([(i, "mmlu", "en") for i in BASE_TASKS if "mmlu" in i])
    type_info.extend([(i, "agieval", "cn") for i in BASE_TASKS if "agieval" in i])
    conn = create_database_table(db_file, table_name, headers, types)

    for data in type_info:
        insert_data(conn, table_name, data)

    conn.commit()
    conn.close()


def delete_table(table_name):
    db_file = DB_NAME[table_name]
    conn = delete_database_table(db_file, table_name)
    conn.commit()
    conn.close()


def create_table(table_name: str):
    headers, db_file = HEADERS[table_name], DB_NAME[table_name]
    types = py_type2db_type(HEADERS[table_name + "_type"])
    assert len(headers) == len(types), "headers and types should have the same length"
    conn = create_database_table(db_file, table_name, headers, types)
    conn.commit()
    conn.close()


def display_data(data, table_name, headers):
    from tabulate import tabulate
    if not data:
        print(f"table {parse_str(table_name)} is empty")
        return data
    else:
        table = tabulate(data, headers=headers, tablefmt="fancy_grid")
        print(table)


def add_db_columns(table_name: str, new_headers: List[str], new_types: List[str]):
    if not isinstance(new_headers, (tuple, list)):
        new_headers = [new_headers]
    if not isinstance(new_types, (tuple, list)):
        new_types = [new_types]
    db_file = DB_NAME[table_name]
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    add_columns_into_table(conn, table_name, new_headers, new_types)
    conn.close()


def read_db_content(table_name, field="all", display=True, **kwargs):
    """fetch and display database table"""
    headers, db_file = HEADERS[table_name], DB_NAME[table_name]
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    data = select_from_db(conn.cursor(), table_name, field=field, **kwargs)
    conn.close()
    if display:
        display_data(data, table_name, headers)
    return data


def update_db_content(table_name, update_dict: Dict, **kwargs):
    """update database table"""
    headers, db_file = HEADERS[table_name], DB_NAME[table_name]
    for k in update_dict:
        assert k in headers, f"{k} is not in {headers}"

    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    update_data(conn, table_name, update_dict, close=True, **kwargs)


def delete_db_content(table_name, **kwargs):
    headers, db_file = HEADERS[table_name], DB_NAME[table_name]
    ensure_del = input(f"Are you sure to delete {table_name} {where_clause_str(**kwargs)}? (y/n)")
    if ensure_del.lower() != "y":
        print("abort")
        return
    else:
        conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
        delete_data(conn, table_name, close=True, **kwargs)
        print("delete successfully")


def check_field(field, headers):
    if isinstance(field, (list, tuple)):
        for x in field:
            assert x in headers, f"{x} is not in {headers}"
    elif field != "all" or field != "*":
        assert field in headers, f"{field} is not in {headers}"


def unpack_data(data):
    if len(data) == 1:
        data = data[0]
    if len(data) == 1:
        data = data[0]
    return data


def get_dataset_info(dataset_name="human_eval", field="all", **kwargs):
    table_name = "datasets_info"
    db_file = DB_NAME[table_name]
    check_field(field, HEADERS[table_name])
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    data = select_from_db(conn.cursor(), table_name, field, dataset=dataset_name, **kwargs)
    conn.commit()
    conn.close()
    return unpack_data(data)


def get_model_info(field="all", **kwargs):
    table_name = "model_info"
    db_file = DB_NAME[table_name]
    check_field(field, HEADERS[table_name])
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    model_info = select_from_db(conn.cursor(), table_name, field, **kwargs)
    conn.commit()
    conn.close()
    return unpack_data(model_info)


def get_topk_modelid(task_name: str, score_name: str = "metric_value", k: int = 20) -> List[int]:
    """get topk metric value of a task like ceval, only max eval_count value considered for the same model id."""
    dataset_id = get_dataset_info(task_name, field=["id"])
    table_name = "evaluate_metrics"
    score_key = "metric_type"
    target_value = "average"
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, DB_NAME[table_name]))
    cursor = conn.cursor()
    cmd = f"""
SELECT t1.*
FROM {table_name} t1
JOIN (
  SELECT model_id, MAX(eval_count) AS max_eval_count
  FROM {table_name}
  GROUP BY model_id
) t2 ON t1.model_id = t2.model_id AND t1.eval_count = t2.max_eval_count AND t1.{score_key} = '{target_value}'
WHERE t1.dataset_id = {dataset_id}
ORDER BY t1.{score_name} DESC
LIMIT {k};
"""
    cursor.execute(cmd)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return [x[1] for x in result]


def get_max_eval_count(model_id, dataset_name="human_eval", by_metrics=False) -> int:
    """If model_id is not in the table, return 0"""
    dataset_id, task_name = get_dataset_info(dataset_name, field=["id", "task_type"])
    if by_metrics:
        task_name = "evaluate_metrics"
    else:
        task_name = task_name + "_eval"

    kwargs = {"model_id": model_id, "dataset_id": dataset_id}
    db_file = DB_NAME[task_name]
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))

    cursor = conn.cursor()
    where_str = where_clause_str(**kwargs)
    query = f"SELECT MAX(eval_count) AS max_eval_count FROM {task_name} {where_str}"
    cursor.execute(query)

    result = cursor.fetchone()
    max_eval_count = result[0] if result[0] is not None else 0

    cursor.close()
    conn.close()

    return max_eval_count


def all_evaluate_model_id() -> List[int]:
    table_name = "evaluate_metrics"
    db_file = DB_NAME[table_name]
    cmd = f"SELECT DISTINCT model_id FROM {table_name}"
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    cursor = conn.cursor()
    cursor.execute(cmd)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return [x[0] for x in result]


def get_metric_values(model_id):
    # ["id", "model_id", "dataset_id", "eval_count", "metric_type", "metric_value"],
    table_name = "evaluate_metrics"
    db_name = DB_NAME[table_name]
    cmd = f"""
    SELECT dataset_id, metric_type, metric_value
    FROM {table_name}
    WHERE model_id = {model_id}
    AND (dataset_id, eval_count) IN (
    SELECT dataset_id, MAX(eval_count)
        FROM {table_name}
        WHERE model_id = {model_id}
        GROUP BY dataset_id
    )
    """
    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_name))
    cursor = conn.cursor()
    cursor.execute(cmd)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return unpack_data(result)


def ensure_model_info_exist(db_file="model_info.sqlite", **kwargs) -> int:
    """ensure model info exists in table and return the id of the model"""

    # model info not exist in table, write it into
    copy_kwargs = kwargs.copy()
    if "topp" not in copy_kwargs or copy_kwargs["topp"] is None:
        copy_kwargs["topp"] = -1
    if "topk" not in copy_kwargs or copy_kwargs["topk"] is None:
        copy_kwargs["topk"] = -1

    model_id = get_model_info(field="id", **copy_kwargs)
    if model_id:
        return model_id

    table_name = "model_info"
    headers = HEADERS[table_name]

    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    insert_values = [copy_kwargs[k] for k in headers if k != "id"]
    insert_data(conn, table_name, insert_values, auto_inc=True, check=True)
    model_id = select_from_db(conn.cursor(), table_name, "id", **copy_kwargs)
    conn.commit()
    conn.close()
    return unpack_data(model_id)


def create_all_database():
    # datasets info table
    fill_datasets_info_table()
    read_db_content("datasets_info")

    # dataset content table
    for dataset_name in BASE_TASKS:
        fill_dataset_table(dataset_name)
    fill_dataset_table("human_eval")
    fill_dataset_table("math")

    tables = [
        "codegen_eval", "math_eval", "generation_eval", "multi_choice_eval",
        "model_info", "evaluate_metrics", "ceval_eval", "bbh_eval", "mmlu_eval", "agieval_eval",
    ]
    for table in tables:
        create_table(table)


def fill_evaluate_metrics(data: List):
    """
    data should be a list of tuples, each tuple is a row of data.
    data format: (model_id, dataset_id, eval_count, metric_type, metric_value)
    """
    # "evaluate_metrics": ["id", "model_id", "dataset_id", "eval_count", "metric_type", "metric_value"],
    table_name = "evaluate_metrics"
    db_file, headers = DB_NAME[table_name], HEADERS[table_name]
    types = py_type2db_type(HEADERS[table_name + "_type"])
    assert len(headers) == len(types), "headers and types should have the same length"

    conn = sqlite3.connect(os.path.join(DATABASE_PATH, db_file))
    for single_data in tqdm(data):
        insert_data(conn, table_name, single_data, check=False)

    conn.commit()
    conn.close()


def metrics_to_database(metrics: Dict[str, float], model_id: int, dataset_id: int, eval_count: int, percentage=True):
    metrics2db = []
    for k, v in metrics.items():
        if percentage:  # metric value like 0.132 -> 13.2
            v = v * 100
        metrics2db.append((model_id, dataset_id, eval_count, k, v))
    fill_evaluate_metrics(metrics2db)


def write_metrics_by_args(args, metrics, dataset, by_metrics=True, percentage=True):
    model_id = ensure_model_info_exist(
            model_name=args.model,
            topk=args.topk,
            topp=args.topp,
            temperature=args.temperature
    )
    dataset_id = get_dataset_info(dataset, field="id")
    eval_count = get_max_eval_count(model_id, dataset_name=dataset, by_metrics=by_metrics) + 1
    metrics_to_database(metrics, model_id, dataset_id, eval_count, percentage=percentage)


def test_evaluate_metrics():
    model_id = 3
    eval_count = 1
    data = [
        (1, "pass@1", 40.2),
        (1, "pass@10", 20.2),
        (1, "pass@100", 10.2),
        (2, "accuracy", 1.4),
        (3, "accuracy", 10.2),
        (4, "accuracy", 10.2),
        (5, "accuracy", 10.2),
        (6, "accuracy", 10.2),
        (8, "accuracy", 10.2),
    ]
    data2db = []
    for d in tqdm(data):
        dataset_id, metric_type, metric_value = d
        data2db.append((model_id, dataset_id, eval_count, metric_type, metric_value))
    fill_evaluate_metrics(data2db)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="read", choices=["create_all", "read", "update", "delete", "merge", "create_table"])
    parser.add_argument("--table", type=str, default="datasets_info", choices=[x for x in HEADERS.keys() if not x.endswith("_type")])
    parser.add_argument(
        "--update_cmd", type=str, default=None,
        help="update command, key-value pair connected by '=', e.g.: model_id=1",
    )
    # the following args are for filters
    parser.add_argument(
        "opts",
        help="Keys to filter database table, e.g. model_id=1 dataset_id=2",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def cmd2dict(cmd: str) -> Dict:
    """convert command to filter"""
    left, right = cmd.split("=")
    try:
        right = int(right)
    except Exception:
        try:
            right = float(right)
        except Exception:
            pass
    return {left: right}


def argopt2filters(opt: List[str]) -> Dict:
    filters = {k: v for x in opt for k, v in cmd2dict(x).items()}
    return filters


@logger.catch
def db_cli():
    args = make_parser().parse_args()
    filters = argopt2filters(args.opts)
    if args.task == "read":
        read_db_content(args.table, **filters)
    elif args.task == "create_all":
        create_all_database()
    elif args.task == "update":
        assert args.update_cmd is not None, "update command should be provided"
        update_dict = cmd2dict(args.update_cmd)
        update_db_content(args.table, update_dict, **filters)
    elif args.task == "delete":
        delete_db_content(args.table, **filters)
    elif args.task == "merge":
        # TODO
        pass
        # merge_table("evaluate_info2.sqlite", "evaluate_info.sqlite", "codegen_eval")
    elif args.task == "create_table":
        create_table(args.table)


if __name__ == "__main__":
    db_cli()
