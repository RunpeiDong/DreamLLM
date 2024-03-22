#!/usr/bin/env python3

import argparse
import streamlit as st
import pandas as pd
from functools import lru_cache, partial
from omni.utils.loguru import logger
from typing import List, Tuple, Dict, Callable
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode
from streamlit_option_menu import option_menu

from llama_evaluation.utils import get_dataset_info, read_db_content, get_metric_values, get_model_info, get_max_eval_count, all_evaluate_model_id, get_topk_modelid, BASE_TASKS

ALL_MODEL_IDs_WITH_METRICS = all_evaluate_model_id()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", action="store_true", help="display info in terminal")
    parser.add_argument("--width", type=int, default=2000, help="width of dataframe")
    parser.add_argument("--height", type=int, default=1000, help="height of dataframe")
    return parser


def aggrid_interactive_table(df: pd.DataFrame, key: str = None, single_select: bool = True, **kwargs):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe
        key (str, optional): Unique key for the widget. Defaults to None.
        single_select (bool, optional): Whether to allow single selection. Defaults to True.
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )
    options.configure_side_bar()
    select_opt = "single" if single_select else "multiple"
    options.configure_selection(select_opt)
    if not single_select:
        options.configure_grid_options(enableRangeSelection=True)

    grid = AgGrid(
        df,
        # fit_columns_on_grid_load=True,
        columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        # update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        reload_data=True,
        key=key,
        **kwargs,
    )
    return grid


def id_transfer(join_on: str) -> Tuple[str]:
    """split id like 'sample_id -> id' to 'sample_id', 'id'"""
    prev, post = join_on.split("->")
    return prev.strip(), post.strip()


def strip_idx_value(strip_idx_str: str) -> List[int]:
    """split strip index like '1,2,3' to [1, 2, 3], If strip index is None, return empty list"""
    if strip_idx_str is None:
        return []
    else:
        return [int(idx) for idx in strip_idx_str.split(",")]


def join_dataframe(
    data_samples: List[Tuple], right: pd.DataFrame, headers: List[str],
    join_str: str = "sample_id -> id", strip_idx_str: str = None,
) -> pd.DataFrame:
    """
    join dataframe with data_samples

    Args:
        data_samples: list of data samples
        right: dataframe to join
        headers (List[str]): headers of data_samples
        join_str: join string, connected by previous key and post key with '->'.
            For example, "sample_id -> id".
        3trip_idx_str: strip index string, connected by ','. For example, "1,2,3".
    """
    if isinstance(data_samples, tuple):  # single sample, warp with list
        data_samples = [data_samples]
    join, on = id_transfer(join_str)
    strip_idx = strip_idx_value(strip_idx_str)
    assert join in right.columns, f"{join} not in {right.columns}"
    for data_idx, data in enumerate(data_samples):
        list_data = list(data)
        for idx in strip_idx:
            list_data[idx] = list_data[idx].strip()
        data_samples[data_idx] = tuple(list_data)
    sample_frame = pd.DataFrame(data_samples, columns=headers)
    joined_df = sample_frame.join(right.set_index(join), on=on)
    return joined_df


def codegen_front_page(model_id, task="human_eval", display=False, width=2000, height=1000):
    """front page of code generation evaluation"""
    fields = ["sample_id", "output", "function", "pass"]
    max_eval_count = get_max_eval_count(model_id, task)
    codegen_data = read_db_content(
        "codegen_eval", field=fields,
        model_id=model_id, display=display, eval_count=max_eval_count,
    )
    code_idx = fields.index("output")
    for data_idx, data in enumerate(codegen_data):
        new_data = list(data)
        new_data[code_idx] = data[code_idx].strip()
        codegen_data[data_idx] = tuple(new_data)
    df = pd.DataFrame(codegen_data, columns=fields)

    data_samples = read_db_content(task, display=False)
    headers = ["id", "question", "test"]
    joined_df = join_dataframe(
        data_samples, df, headers,
        join_str="sample_id -> id", strip_idx_str="1,2"
    )
    st.dataframe(joined_df, width=width, height=height)


def aligned_text(text: str, color="black", size=1):
    st.markdown(f"<h{size} style='text-align: center; color: {color};'>{text}</h{size}>", unsafe_allow_html=True)


def unified_front_page(model_id, task, dataset, width=2000, height=1000, display=False):
    """front page of generation evaluation"""
    if task == "Generation":
        fields = ["sample_id", "raw_output","output", "exact_match", "include"]
        headers = ["id", "question", "labels"]
        table_name = "generation_eval"
    elif task == "Multi-choice":
        fields = ["sample_id", "output", "pass"]
        headers = ["id", "question", "options", "label"]
        table_name = "multi_choice_eval"
    elif task == "Math":
        fields = ["sample_id", "output", "short_answer", "pass"]
        headers = ["id", "problem", "level", "solution", "short_solution"]
        table_name = "math_eval"
    elif task == "CEval":
        fields = ["sample_id", "output", "logprobs", "pass"]
        headers = ["id", "question", "labels"]
        table_name = "ceval_eval"
    elif task == "MMLU":
        fields = ["sample_id", "output", "logprobs", "pass"]
        headers = ["id", "question", "labels"]
        table_name = "mmlu_eval"
    elif task == "AGIEval":
        fields = ["sample_id", "output", "logprobs", "pass"]
        headers = ["id", "question", "labels"]
        table_name = "agieval_eval"

    dataset_id = get_dataset_info(dataset, "id")
    max_eval_count = get_max_eval_count(model_id, dataset)
    gen_data = read_db_content(
        table_name, field=fields, display=display,
        model_id=model_id, dataset_id=dataset_id, eval_count=max_eval_count,
    )
    df = pd.DataFrame(gen_data, columns=fields)

    data_samples = read_db_content(dataset, field=headers, display=display)
    joined_df = join_dataframe(
        data_samples, df, headers,
        join_str="sample_id -> id", strip_idx_str=None,
    )
    st.dataframe(joined_df, width=width, height=height)


def display_model_info(key="tab1", single_select: bool = True, model_filter: str = None):
    headers = ["id", "model_name", "temperature", "topp", "topk"]
    aligned_text("需要查看哪个模型的推理结果？双击模型信息即可查看。", size=5)
    models_info = read_db_content("model_info", field=headers, display=False)
    if model_filter is not None:
        models_info = list(filter(model_filter, models_info))
    model_df = pd.DataFrame(models_info, columns=headers)
    grid = aggrid_interactive_table(model_df, key=key, single_select=single_select, height=200)
    return grid


def display_dataset(option):
    select_text = "需要查看哪个数据集的推理结果？请点击选择。"
    if option == "Codegen":
        dataset_name = st.selectbox(select_text, ("human_eval",))
    elif option == "Math":
        dataset_name = st.selectbox(select_text, ("math", ))
    elif option == "Multi-choice":
        choices = ("boolq", "piqa", "siqa", "hellaswag", "winogrande", "arc_e", "arc_c", "obqa", "race_m", "race_h", "clue_c3", "clue_wsc")  # noqa
        dataset_name = st.selectbox(select_text, choices)
    elif option == "Generation":
        dataset_name = st.selectbox(select_text, ("triviaqa", "naturalqa", "drop_gen", "clue_cmrc"))
    elif option == "CEval" or option == "MMLU" or option == "AGIEval":
        task = option.lower()
        table = tuple(i for i in BASE_TASKS if task in i)
        dataset_name = st.selectbox(select_text, table)
    else:
        raise ValueError(f"Unknown option {option}")
    return dataset_name


def display_inference_result(model_id, args):
    task_type = st.selectbox(
        '需要查看哪一个任务的结果？请点击选择',
        ("Codegen", "Multi-choice", "Generation", "Math", "CEval", "MMLU", "AGIEval"),
    )
    dataset_name = display_dataset(task_type)
    kwargs = {"display": args.display, "width": args.width, "height": args.height}
    aligned_text(dataset_name + " result", size=4)
    if dataset_name == "human_eval":
        codegen_front_page(model_id, **kwargs)
    else:
        unified_front_page(model_id, task_type, dataset_name, **kwargs)


@lru_cache(maxsize=1)
def dataset_info() -> pd.DataFrame:
    headers = ["id", "dataset"]
    data_info = read_db_content("datasets_info", field=headers, display=False)
    return pd.DataFrame(data_info, columns=headers)


def filter_by_metrics(item, datasets=None):
    if datasets is None:
        return True
    max_eval_count = 0
    for dataset in datasets:
        max_eval_count = max(max_eval_count, get_max_eval_count(item[0], dataset, by_metrics=True))
    return max_eval_count > 0


def get_llamaeval_keys():
    choice_qa_keys = [
        # choice QA
        "boolq/accuracy",
        "piqa/accuracy",
        "siqa/accuracy",
        "hellaswag/accuracy",
        "winogrande/accuracy",
        "arc_e/accuracy",
        "arc_c/accuracy",
        "obqa/accuracy",
    ]
    qa_avg_keyname = "QA_Average_accuracy"
    wanted_metric = [
        "clue_c3/accuracy",
        "clue_cmrc/include",
        "clue_wsc/accuracy",
        "clue_cmrc/include",
        *choice_qa_keys,
        qa_avg_keyname,
        # comprehension QA
        "race_m/accuracy",
        "race_h/accuracy",
        # generation QA
        "naturalqa/include",
        "triviaqa/include",
        "drop_gen/include",
        # code math
        "human_eval/pass@1",
        "math/overall_acc",
    ]
    return wanted_metric, [(choice_qa_keys, qa_avg_keyname)]


def get_falconsmall_keys():
    en_qa_keys = [
        "boolq/accuracy",
        "piqa/accuracy",
        "hellaswag/accuracy",
        "winogrande/accuracy",
        "arc_e/accuracy",
        "sciq/accuracy",
    ]
    zh_qa_keys = [
        "clue_c3/accuracy",
        "clue_cmrc/include",
        "xtreme/include",
    ]
    en_avg_keyname = "EN_Average_accuracy"
    zh_avg_keyname = "ZH_Average_accuracy"
    wanted_metric = [
        zh_avg_keyname,
        en_avg_keyname,
        *zh_qa_keys,
        *en_qa_keys,
        "race_m/accuracy",
        "triviaqa/include",
        "drop_gen/include",
    ]
    return wanted_metric, [(en_qa_keys, en_avg_keyname), (zh_qa_keys, zh_avg_keyname)]


CEAVAL_KEYS = [
    "ceval/stem",
    "ceval/social science",
    "ceval/humanities",
    "ceval/other",
    "ceval/hard",
    "ceval/average",
]


MMLU_KEYS = [
    "mmlu/stem",
    "mmlu/humanities",
    "mmlu/social sciences",
    "mmlu/other (business, health, misc.)",
    "mmlu/average",
]


AGIEVAL_KEYS = [
    "agieval/math",
    "agieval/logiqa-en",
    "agieval/logiqa-zh",
    "agieval/lsat-ar",
    "agieval/lsat-lr",
    "agieval/lsat-rc",
    "agieval/sat-math",
    "agieval/sat-en",
    "agieval/sat-en-without-passage",
    "agieval/gaokao-chinese",
    "agieval/gaokao-english",
    "agieval/gaokao-geography",
    "agieval/gaokao-history",
    "agieval/gaokao-biology",
    "agieval/gaokao-chemistry",
    "agieval/gaokao-physics",
    "agieval/gaokao-mathqa",
    "agieval/gaokao-mathcloze",
]


def get_cevalmmlu_keys():
    wanted_metric = [*CEAVAL_KEYS, *MMLU_KEYS]
    return wanted_metric, []


def get_agieval_keys():
    avg_keyname = "agieval/gaokao-Average"
    wanted_metric = [*AGIEVAL_KEYS, avg_keyname]
    avg_keys = wanted_metric[-10:-1]
    return wanted_metric, [(avg_keys, avg_keyname)]


def metrics_checkbox() -> Dict[str, bool]:
    """sidebar checkbox for metrics"""
    st.sidebar.write("需要查看哪些指标？请点击选择（可多选）。")
    llama_eval = st.sidebar.checkbox("LLaMAEval", value=True)
    falcon_small = st.sidebar.checkbox("Falcon-S")
    ceval_mmlu = st.sidebar.checkbox("Ceval MMLU")
    agi_eval = st.sidebar.checkbox("AGIEval")
    return {
        "llama_eval": llama_eval,
        "falcon_small": falcon_small,
        "ceval_mmlu": ceval_mmlu,
        "agieval": agi_eval,
    }


SHOWN_DICT = {
    "llama_eval": get_llamaeval_keys(),
    "falcon_small": get_falconsmall_keys(),
    "ceval_mmlu": get_cevalmmlu_keys(),
    "agieval": get_agieval_keys(),
}

MODEL_FILTER = {
    "llama_eval": lambda x: x[0] in ALL_MODEL_IDs_WITH_METRICS,
    "falcon_small": lambda x: '1b' in x[1] or '1dot5b' in x[1],
    "ceval_mmlu": lambda x: filter_by_metrics(x, datasets=["ceval", "mmlu"]),
    "agieval": lambda x: filter_by_metrics(x, datasets=["agieval"]),
}


def metrics_result(
    model_id: int, filter_df: bool = True,
    wanted_metric: List[str] = None, avg_items_list: List[str] = None
) -> pd.DataFrame:
    """given a model id, return a dataframe with all wanted metrics of this model

    Args:
        model_id (int): model id
        filter_df (bool, optional): whether to filter the dataframe. Defaults to True.
    """
    if wanted_metric is not None and avg_items_list is None:
        avg_items_list = []
    if wanted_metric is None:
        wanted_metric, avg_items_list = [], []
        filter_keys = [key for key, status in st.session_state.checkbox_status.items() if status]
        for key in filter_keys:
            metric, avg_items = SHOWN_DICT[key]
            wanted_metric.extend(metric)
            avg_items_list.extend(avg_items)

    df = dataset_info()
    model_name = get_model_info(field="model_name", id=model_id)
    value_headers = ["dataset_id", "metric", model_name]
    values = get_metric_values(model_id)
    joined_df = join_dataframe(values, df, value_headers, join_str="id -> dataset_id")
    joined_df["metric"] = joined_df["dataset"] + "/" + joined_df["metric"]
    joined_df.rename(columns={"value": "metric_value"}, inplace=True)
    joined_df = joined_df.drop(columns=["dataset_id", "dataset"])
    if filter_df:
        mask = joined_df["metric"].isin(wanted_metric)
        joined_df = joined_df[mask]
        joined_df = joined_df.drop_duplicates("metric", keep="last")
        joined_df = sorted_by_keys(joined_df, wanted_metric)
        for item in avg_items_list:
            joined_df = average_by_keys(joined_df, *item)
        joined_df = sorted_by_keys(joined_df, wanted_metric)

    for col in joined_df.columns[1:]:
        joined_df[col] = joined_df[col].round(decimals=1)
    return joined_df


def sorted_by_keys(df: pd.DataFrame, keys_to_sort: List[str]) -> pd.DataFrame:
    """sort the dataframe by values in keys_to_sort"""
    order_dict = {value: index for index, value in enumerate(keys_to_sort)}
    df['order'] = df['metric'].map(order_dict)
    df = df.sort_values('order')
    df = df.drop('order', axis=1)
    df = df.reset_index(drop=True)
    return df


def average_by_keys(df: pd.DataFrame, keys_to_average: List[str], avg_keyname: str) -> pd.DataFrame:
    """average the values of keys_to_average, and add a new row with avg_keyname"""
    mask = df['metric'].isin(keys_to_average)
    if mask.sum() == 0:
        return df
    avg = df[mask].mean(numeric_only=True).values[0]
    df.loc[len(df)] = [avg_keyname, avg]
    return df


def highlight_max(row, color="yellow"):
    """highlight the maximum in a Series color."""
    max_value = row[1:].max()
    return [f"background-color: {color}" if v == max_value else '' for v in row]


def add_metrics_column(df: pd.DataFrame) -> pd.DataFrame:
    st.session_state.data = pd.merge(st.session_state.data, df, on="metric", how="outer")


def reset_metrics_page():
    st.session_state.data = None
    st.session_state.model_id = []


def visualize_page(args):
    grid = display_model_info(key="tab1")
    selected = grid["selected_rows"]
    if selected:  # select one model here
        selected: Dict = selected[0]
        if "_selectedRowNodeInfo" in selected:
            selected.pop("_selectedRowNodeInfo")
        model_id = selected["id"]
        aligned_text(f"模型信息: model_id={model_id},  model_name={selected['model_name']}", color="red", size=5)
        display_inference_result(model_id, args)


def filter_function(checkbox_states) -> Callable:
    def join_f(func_list, x):
        for func in func_list:
            if func(x):
                return True
        return False

    filter_funcs = [MODEL_FILTER[key] for key, status in checkbox_states.items() if status]
    join_filter = partial(join_f, filter_funcs)
    return join_filter


def update_checkbox_status(checkbox_state) -> bool:
    update = False
    if "checkbox_status" not in st.session_state:
        st.session_state["checkbox_status"] = checkbox_state
    if st.session_state.checkbox_status != checkbox_state:
        reset_metrics_page()
        update = True
        st.session_state["checkbox_status"] = checkbox_state
    return update


def update_dataframe_status(model_id: int, joined_df: pd.DataFrame):
    # add model_id to session anyway
    has_seen_model = True
    if "model_id" not in st.session_state or st.session_state.model_id is None:
        has_seen_model = False
        st.session_state.model_id = [model_id]
    else:
        if model_id not in st.session_state.model_id:
            has_seen_model = False
            st.session_state.model_id.append(model_id)

    # check if empty dataframe
    if len(joined_df) == 0:
        st.warning("该模型没有任何评测结果")
    else:
        if "data" not in st.session_state or st.session_state.data is None:  # first time to show
            st.session_state.data = joined_df
        else:
            if not has_seen_model:
                add_metrics_column(joined_df)


def benchmark_table_page(table_id="tab2", model_filter=None):
    st.sidebar.markdown("## 控制面板")
    checkbox_state = metrics_checkbox()
    update = update_checkbox_status(checkbox_state)

    model_filter = filter_function(st.session_state.checkbox_status)

    grid = display_model_info(key=table_id, model_filter=model_filter)
    selected = grid["selected_rows"] if not update else False

    if selected:  # select one model here
        selected: Dict = selected[0]
        if "_selectedRowNodeInfo" in selected:
            selected.pop("_selectedRowNodeInfo")
        model_id = selected["id"]
        aligned_text(f"模型信息: model_id={model_id},  model_name={selected['model_name']}", color="red", size=5)

        joined_df = metrics_result(model_id)
        update_dataframe_status(model_id, joined_df)

        st.sidebar.warning("每个评测指标的最大值用黄色高亮")
        if st.sidebar.button("reset"):
            reset_metrics_page()
        st.sidebar.warning(f"展示的模型id: {st.session_state.model_id}")
        # st.sidebar.warning(f"checkbox状态: {st.session_state.checkbox_status}")

        if st.session_state.data is not None:
            height = (min(len(st.session_state.data), 25) + 1) * 35
            st.dataframe(
                st.session_state.data.style.apply(highlight_max, axis=1).format(
                    {col: '{:.1f}' for col in st.session_state.data.columns[1:]}
                ), height=height,
            )


def leaderboard_page(k_default: int = 10):
    # _, left_selectbox, _, right_area, _ = st.columns(5)
    select_text = "需要查看哪个任务的Leaderboard？请点击选择。"
    option = st.selectbox(select_text, ("ceval", "mmlu", "agi-eval"))
    k_value = st.text_area(f"请在此输入查看的最多模型数，默认{k_default}。", value=k_default)
    st.write(f"k_value: {k_value}")
    try:
        k_value = int(k_value)
    except ValueError:
        st.warning(f"k_value必须是整数，已经设置为默认值{k_default}")
        k_value = k_default

    if option not in ("ceval", "mmlu", "agi-eval"):
        raise ValueError(f"option must be one of ceval, mmlu, agi-eval, but got {option}")

    st.write(f"{option.upper()} Leaderboard")
    wanted_metric = {
        "ceval": CEAVAL_KEYS,
        "mmlu": MMLU_KEYS,
        "agi-eval": AGIEVAL_KEYS,
    }[option]
    task_name = "agieval" if option == "agi-eval" else option
    model_ids = get_topk_modelid(task_name, k=k_value)
    st.warning(f"model_id: {model_ids}")
    for idx, model_id in enumerate(model_ids):
        if idx == 0:
            joined_df = metrics_result(model_id, wanted_metric=wanted_metric, avg_items_list=None)
        else:
            df = metrics_result(model_id, wanted_metric=wanted_metric, avg_items_list=None)
            joined_df = pd.merge(joined_df, df, on="metric", how="outer")
    st.dataframe(joined_df.T, use_container_width=True)


@logger.catch
def main_page():
    args = make_parser().parse_args()
    st.set_page_config(page_title="LLM benchmark", layout="wide")

    options = ["推理结果", "评测结果", "Leaderboard"]
    icons = ["gear", "list-task", "list-task"]
    page_opt = option_menu(
        None,
        options,
        icons=icons,
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        key="page_option",
    )
    if page_opt == options[0]:
        visualize_page(args)
    elif page_opt == options[1]:
        benchmark_table_page()
    elif page_opt == options[2]:
        leaderboard_page()


if __name__ == "__main__":
    # `streamlit run fe_display.py` to execute the app
    main_page()
